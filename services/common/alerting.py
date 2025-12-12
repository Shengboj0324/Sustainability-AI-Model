"""
Alerting System

CRITICAL: Production-grade alerting for critical failures and anomalies
- Multiple notification channels (email, Slack, PagerDuty, webhooks)
- Alert rules and thresholds
- Alert deduplication and grouping
- Escalation policies
- Alert history and tracking
- Rate limiting to prevent alert storms

Features:
- Email notifications (SMTP)
- Slack notifications (webhooks)
- PagerDuty integration
- Custom webhooks
- Alert severity levels
- Alert deduplication (time-based)
- Alert rate limiting
- Alert history
- Async notification delivery

Usage:
    from services.common.alerting import AlertManager, Alert, AlertSeverity
    
    # Initialize alert manager
    alert_manager = AlertManager(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        smtp_user="alerts@example.com",
        smtp_password="...",
        slack_webhook="https://hooks.slack.com/...",
        pagerduty_key="..."
    )
    
    # Send alert
    await alert_manager.send_alert(Alert(
        title="Circuit Breaker Opened",
        message="Neo4j circuit breaker opened after 5 failures",
        severity=AlertSeverity.CRITICAL,
        service="kg_service",
        tags={"component": "neo4j", "circuit_breaker": "open"}
    ))
"""

import os
import logging
import asyncio
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import httpx

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Alert data structure
    
    Attributes:
        title: Alert title
        message: Alert message
        severity: Alert severity
        service: Service name
        tags: Additional tags for filtering
        timestamp: Alert timestamp (auto-generated)
        fingerprint: Alert fingerprint for deduplication (auto-generated)
    """
    title: str
    message: str
    severity: AlertSeverity
    service: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        """Generate fingerprint if not provided"""
        if self.fingerprint is None:
            # Generate fingerprint from title, service, and tags
            fingerprint_data = f"{self.title}:{self.service}:{sorted(self.tags.items())}"
            self.fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()


class AlertManager:
    """
    Alert manager for sending notifications
    
    Features:
    - Multiple notification channels
    - Alert deduplication
    - Rate limiting
    - Async delivery
    - Error handling
    """
    
    def __init__(
        self,
        # Email configuration
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        smtp_from: Optional[str] = None,
        smtp_to: Optional[List[str]] = None,
        # Slack configuration
        slack_webhook: Optional[str] = None,
        slack_channel: Optional[str] = None,
        # PagerDuty configuration
        pagerduty_key: Optional[str] = None,
        # Webhook configuration
        webhook_url: Optional[str] = None,
        # Deduplication settings
        dedup_window_seconds: int = 300,  # 5 minutes
        # Rate limiting settings
        max_alerts_per_minute: int = 10,
        max_alerts_per_hour: int = 100
    ):
        """
        Initialize alert manager
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            smtp_from: From email address
            smtp_to: List of recipient email addresses
            slack_webhook: Slack webhook URL
            slack_channel: Slack channel (optional, overrides webhook default)
            pagerduty_key: PagerDuty integration key
            webhook_url: Custom webhook URL
            dedup_window_seconds: Deduplication window in seconds
            max_alerts_per_minute: Maximum alerts per minute
            max_alerts_per_hour: Maximum alerts per hour
        """
        # Email configuration
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.smtp_from = smtp_from or os.getenv("SMTP_FROM", self.smtp_user)
        self.smtp_to = smtp_to or (os.getenv("SMTP_TO", "").split(",") if os.getenv("SMTP_TO") else [])
        
        # Slack configuration
        self.slack_webhook = slack_webhook or os.getenv("SLACK_WEBHOOK")
        self.slack_channel = slack_channel or os.getenv("SLACK_CHANNEL")

        # PagerDuty configuration
        self.pagerduty_key = pagerduty_key or os.getenv("PAGERDUTY_KEY")

        # Webhook configuration
        self.webhook_url = webhook_url or os.getenv("ALERT_WEBHOOK_URL")

        # Deduplication
        self.dedup_window_seconds = dedup_window_seconds
        self.recent_alerts: Dict[str, datetime] = {}  # fingerprint -> timestamp

        # Rate limiting
        self.max_alerts_per_minute = max_alerts_per_minute
        self.max_alerts_per_hour = max_alerts_per_hour
        self.alerts_last_minute: List[datetime] = []
        self.alerts_last_hour: List[datetime] = []

        # HTTP client for webhooks
        self.http_client = httpx.AsyncClient(timeout=10.0)

        logger.info("Alert manager initialized")

    async def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through all configured channels

        Args:
            alert: Alert to send

        Returns:
            True if alert sent successfully through at least one channel

        Example:
            success = await alert_manager.send_alert(Alert(
                title="High Error Rate",
                message="Error rate exceeded 5% in last 5 minutes",
                severity=AlertSeverity.ERROR,
                service="llm_service",
                tags={"component": "inference"}
            ))
        """
        # Check deduplication
        if self._is_duplicate(alert):
            logger.debug(f"Alert deduplicated: {alert.title}")
            return False

        # Check rate limiting
        if self._is_rate_limited():
            logger.warning(f"Alert rate limited: {alert.title}")
            return False

        # Record alert
        self._record_alert(alert)

        # Send through all channels
        results = await asyncio.gather(
            self._send_email(alert),
            self._send_slack(alert),
            self._send_pagerduty(alert),
            self._send_webhook(alert),
            return_exceptions=True
        )

        # Check if at least one channel succeeded
        success = any(r is True for r in results)

        if success:
            logger.info(f"Alert sent: {alert.title} (severity: {alert.severity})")
        else:
            logger.error(f"Failed to send alert: {alert.title}")

        return success

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within dedup window"""
        if alert.fingerprint in self.recent_alerts:
            last_sent = self.recent_alerts[alert.fingerprint]
            if datetime.utcnow() - last_sent < timedelta(seconds=self.dedup_window_seconds):
                return True
        return False

    def _is_rate_limited(self) -> bool:
        """Check if alert should be rate limited"""
        now = datetime.utcnow()

        # Clean old entries
        self.alerts_last_minute = [t for t in self.alerts_last_minute if now - t < timedelta(minutes=1)]
        self.alerts_last_hour = [t for t in self.alerts_last_hour if now - t < timedelta(hours=1)]

        # Check limits
        if len(self.alerts_last_minute) >= self.max_alerts_per_minute:
            return True
        if len(self.alerts_last_hour) >= self.max_alerts_per_hour:
            return True

        return False

    def _record_alert(self, alert: Alert):
        """Record alert for deduplication and rate limiting"""
        now = datetime.utcnow()
        self.recent_alerts[alert.fingerprint] = now
        self.alerts_last_minute.append(now)
        self.alerts_last_hour.append(now)

    async def _send_email(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not all([self.smtp_host, self.smtp_user, self.smtp_password, self.smtp_to]):
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_from
            msg['To'] = ', '.join(self.smtp_to)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"

            # Create body
            body = f"""
Alert: {alert.title}
Severity: {alert.severity.upper()}
Service: {alert.service}
Time: {alert.timestamp.isoformat()}

Message:
{alert.message}

Tags:
{json.dumps(alert.tags, indent=2)}
"""
            msg.attach(MIMEText(body, 'plain'))

            # Send email (in thread pool to avoid blocking)
            await asyncio.to_thread(self._send_email_sync, msg)

            logger.debug(f"Email sent for alert: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False

    def _send_email_sync(self, msg: MIMEMultipart):
        """Send email synchronously"""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)

    async def _send_slack(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        if not self.slack_webhook:
            return False

        try:
            # Map severity to color
            color_map = {
                AlertSeverity.DEBUG: "#808080",
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }

            # Create Slack message
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#808080"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.upper(), "short": True},
                        {"title": "Service", "value": alert.service, "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": False}
                    ] + [
                        {"title": k, "value": v, "short": True}
                        for k, v in alert.tags.items()
                    ],
                    "footer": "ReleAF AI Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            if self.slack_channel:
                payload["channel"] = self.slack_channel

            # Send to Slack
            response = await self.http_client.post(
                self.slack_webhook,
                json=payload
            )
            response.raise_for_status()

            logger.debug(f"Slack notification sent for alert: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    async def _send_pagerduty(self, alert: Alert) -> bool:
        """Send alert to PagerDuty"""
        if not self.pagerduty_key:
            return False

        # Only send ERROR and CRITICAL alerts to PagerDuty
        if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return False

        try:
            # Map severity to PagerDuty severity
            pd_severity = "critical" if alert.severity == AlertSeverity.CRITICAL else "error"

            # Create PagerDuty event
            payload = {
                "routing_key": self.pagerduty_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.title,
                    "severity": pd_severity,
                    "source": alert.service,
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": {
                        "message": alert.message,
                        "tags": alert.tags
                    }
                }
            }

            # Send to PagerDuty
            response = await self.http_client.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload
            )
            response.raise_for_status()

            logger.debug(f"PagerDuty alert sent: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False

    async def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to custom webhook"""
        if not self.webhook_url:
            return False

        try:
            # Create webhook payload
            payload = {
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "service": alert.service,
                "tags": alert.tags,
                "timestamp": alert.timestamp.isoformat(),
                "fingerprint": alert.fingerprint
            }

            # Send to webhook
            response = await self.http_client.post(
                self.webhook_url,
                json=payload
            )
            response.raise_for_status()

            logger.debug(f"Webhook notification sent for alert: {alert.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def init_alerting(
    smtp_host: Optional[str] = None,
    smtp_port: int = 587,
    smtp_user: Optional[str] = None,
    smtp_password: Optional[str] = None,
    smtp_from: Optional[str] = None,
    smtp_to: Optional[List[str]] = None,
    slack_webhook: Optional[str] = None,
    slack_channel: Optional[str] = None,
    pagerduty_key: Optional[str] = None,
    webhook_url: Optional[str] = None
) -> AlertManager:
    """
    Initialize global alert manager

    Args:
        smtp_host: SMTP server host
        smtp_port: SMTP server port
        smtp_user: SMTP username
        smtp_password: SMTP password
        smtp_from: From email address
        smtp_to: List of recipient email addresses
        slack_webhook: Slack webhook URL
        slack_channel: Slack channel
        pagerduty_key: PagerDuty integration key
        webhook_url: Custom webhook URL

    Returns:
        AlertManager instance

    Example:
        alert_manager = init_alerting(
            slack_webhook=os.getenv("SLACK_WEBHOOK"),
            pagerduty_key=os.getenv("PAGERDUTY_KEY")
        )
    """
    global _alert_manager

    _alert_manager = AlertManager(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_user=smtp_user,
        smtp_password=smtp_password,
        smtp_from=smtp_from,
        smtp_to=smtp_to,
        slack_webhook=slack_webhook,
        slack_channel=slack_channel,
        pagerduty_key=pagerduty_key,
        webhook_url=webhook_url
    )

    return _alert_manager


def get_alert_manager() -> Optional[AlertManager]:
    """Get global alert manager instance"""
    return _alert_manager


async def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity,
    service: str,
    **tags
) -> bool:
    """
    Send alert using global alert manager

    Args:
        title: Alert title
        message: Alert message
        severity: Alert severity
        service: Service name
        **tags: Additional tags

    Returns:
        True if alert sent successfully

    Example:
        await send_alert(
            title="High Error Rate",
            message="Error rate exceeded 5%",
            severity=AlertSeverity.ERROR,
            service="llm_service",
            component="inference"
        )
    """
    if _alert_manager is None:
        logger.warning("Alert manager not initialized")
        return False

    alert = Alert(
        title=title,
        message=message,
        severity=severity,
        service=service,
        tags=tags
    )

    return await _alert_manager.send_alert(alert)

