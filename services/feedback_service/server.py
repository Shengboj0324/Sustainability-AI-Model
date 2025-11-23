"""
Feedback Service - User Feedback Collection & Continuous Improvement

CRITICAL FEATURES:
- User feedback collection (thumbs up/down, ratings, comments)
- Answer quality tracking and analytics
- Feedback storage in PostgreSQL
- Real-time metrics and dashboards
- Automated model retraining triggers
- A/B testing framework
- User satisfaction tracking
- Continuous improvement pipeline

This addresses the requirement for "continuously self improving with users' input data"
"""

import asyncio
import logging
import os
import sys
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
FEEDBACK_TOTAL = Counter('feedback_total', 'Total feedback submissions', ['feedback_type', 'rating'])
FEEDBACK_DURATION = Histogram('feedback_duration_seconds', 'Feedback processing duration')
ACTIVE_REQUESTS = Gauge('feedback_active_requests', 'Active feedback requests')
SATISFACTION_SCORE = Gauge('user_satisfaction_score', 'Average user satisfaction score')
RETRAINING_TRIGGERS = Counter('retraining_triggers_total', 'Model retraining triggers', ['service'])

# Initialize FastAPI app
app = FastAPI(
    title="ReleAF AI Feedback Service",
    description="User feedback collection and continuous improvement system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for web and iOS clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackType(str, Enum):
    """Feedback types"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    COMMENT = "comment"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"


class ServiceType(str, Enum):
    """Service types for feedback"""
    LLM = "llm"
    VISION = "vision"
    RAG = "rag"
    KG = "kg"
    ORCHESTRATOR = "orchestrator"
    OVERALL = "overall"


class FeedbackRequest(BaseModel):
    """User feedback submission"""
    feedback_type: FeedbackType = Field(..., description="Type of feedback")
    service: ServiceType = Field(..., description="Service being rated")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating (1-5 stars)")
    comment: Optional[str] = Field(None, max_length=2000, description="User comment")

    # Context
    query: Optional[str] = Field(None, max_length=1000, description="Original user query")
    response: Optional[str] = Field(None, max_length=5000, description="System response")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    user_id: Optional[str] = Field(None, description="User ID (anonymous hash)")

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('rating')
    def validate_rating(cls, v, values):
        """Validate rating is provided for RATING feedback type"""
        if values.get('feedback_type') == FeedbackType.RATING and v is None:
            raise ValueError("Rating must be provided for RATING feedback type")
        return v


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    feedback_id: str
    status: str
    message: str
    timestamp: datetime


class FeedbackAnalytics(BaseModel):
    """Feedback analytics response"""
    total_feedback: int
    average_rating: float
    satisfaction_rate: float  # % of positive feedback
    feedback_by_type: Dict[str, int]
    feedback_by_service: Dict[str, int]
    recent_comments: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    retraining_recommendations: List[Dict[str, Any]]


class FeedbackService:
    """
    Production-grade feedback service with PostgreSQL storage

    CRITICAL: Enables continuous improvement through user feedback
    """

    def __init__(self):
        self.db_pool: Optional[asyncpg.Pool] = None
        self._shutdown = False

        # Thresholds for retraining triggers
        self.retraining_thresholds = {
            "min_feedback_count": int(os.getenv("RETRAINING_MIN_FEEDBACK", "100")),
            "low_satisfaction_threshold": float(os.getenv("LOW_SATISFACTION_THRESHOLD", "0.6")),
            "min_negative_feedback": int(os.getenv("MIN_NEGATIVE_FEEDBACK", "20"))
        }

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            logger.info("Initializing feedback service...")

            # Create connection pool
            self.db_pool = await asyncpg.create_pool(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                database=os.getenv("POSTGRES_DB", "releaf_feedback"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Create tables if not exist
            await self._create_tables()

            logger.info("Feedback service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize feedback service: {e}", exc_info=True)
            raise

    async def _create_tables(self):
        """Create database tables"""
        async with self.db_pool.acquire() as conn:
            # Feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    feedback_id VARCHAR(64) UNIQUE NOT NULL,
                    feedback_type VARCHAR(50) NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    rating INTEGER,
                    comment TEXT,
                    query TEXT,
                    response TEXT,
                    session_id VARCHAR(128),
                    user_id VARCHAR(128),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    processed BOOLEAN DEFAULT FALSE,
                    processed_at TIMESTAMP
                )
            """)

            # Create indices
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_feedback_service ON feedback(service);
                CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
                CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating);
                CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback(processed);
            """)

            # Retraining triggers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    id SERIAL PRIMARY KEY,
                    service VARCHAR(50) NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    feedback_count INTEGER,
                    satisfaction_score FLOAT,
                    negative_feedback_count INTEGER,
                    triggered_at TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'pending',
                    completed_at TIMESTAMP
                )
            """)

            logger.info("Database tables created/verified")

    async def close(self):
        """Graceful shutdown"""
        try:
            self._shutdown = True
            logger.info("Shutting down feedback service...")

            if self.db_pool:
                await self.db_pool.close()
                logger.info("Database pool closed")

            logger.info("Feedback service shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)

    async def submit_feedback(self, feedback: FeedbackRequest) -> Dict[str, Any]:
        """
        Submit user feedback

        CRITICAL: Stores feedback and triggers retraining if thresholds met
        """
        try:
            # Generate feedback ID
            import hashlib
            feedback_id = hashlib.sha256(
                f"{feedback.service}_{feedback.feedback_type}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO feedback (
                        feedback_id, feedback_type, service, rating, comment,
                        query, response, session_id, user_id, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    feedback_id, feedback.feedback_type.value, feedback.service.value,
                    feedback.rating, feedback.comment, feedback.query, feedback.response,
                    feedback.session_id, feedback.user_id, json.dumps(feedback.metadata)
                )

            # Update metrics
            FEEDBACK_TOTAL.labels(
                feedback_type=feedback.feedback_type.value,
                rating=str(feedback.rating) if feedback.rating else "none"
            ).inc()

            # Check if retraining should be triggered
            await self._check_retraining_trigger(feedback.service.value)

            # Update satisfaction gauge
            await self._update_satisfaction_metrics()

            logger.info(f"Feedback submitted: {feedback_id} (service={feedback.service.value}, type={feedback.feedback_type.value})")

            return {
                "feedback_id": feedback_id,
                "status": "success",
                "message": "Thank you for your feedback!",
                "timestamp": datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}", exc_info=True)
            raise

    async def _check_retraining_trigger(self, service: str):
        """
        Check if retraining should be triggered based on feedback

        CRITICAL: Automated continuous improvement trigger
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent feedback stats (last 7 days)
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_feedback,
                        AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE 0 END) as avg_rating,
                        COUNT(CASE WHEN feedback_type IN ('thumbs_down', 'bug_report') THEN 1 END) as negative_count,
                        COUNT(CASE WHEN feedback_type = 'thumbs_up' THEN 1 END) as positive_count
                    FROM feedback
                    WHERE service = $1
                    AND created_at > NOW() - INTERVAL '7 days'
                """, service)

                total_feedback = stats['total_feedback']
                avg_rating = float(stats['avg_rating']) if stats['avg_rating'] else 0
                negative_count = stats['negative_count']
                positive_count = stats['positive_count']

                # Calculate satisfaction rate
                satisfaction_rate = positive_count / total_feedback if total_feedback > 0 else 1.0

                # Check thresholds
                should_trigger = False
                trigger_reason = []

                if total_feedback >= self.retraining_thresholds["min_feedback_count"]:
                    if satisfaction_rate < self.retraining_thresholds["low_satisfaction_threshold"]:
                        should_trigger = True
                        trigger_reason.append(f"Low satisfaction rate: {satisfaction_rate:.2%}")

                    if negative_count >= self.retraining_thresholds["min_negative_feedback"]:
                        should_trigger = True
                        trigger_reason.append(f"High negative feedback: {negative_count} reports")

                    if avg_rating > 0 and avg_rating < 3.0:
                        should_trigger = True
                        trigger_reason.append(f"Low average rating: {avg_rating:.2f}/5.0")

                # Trigger retraining if needed
                if should_trigger:
                    await conn.execute("""
                        INSERT INTO retraining_triggers (
                            service, trigger_reason, feedback_count,
                            satisfaction_score, negative_feedback_count
                        ) VALUES ($1, $2, $3, $4, $5)
                    """, service, "; ".join(trigger_reason), total_feedback,
                        satisfaction_rate, negative_count)

                    RETRAINING_TRIGGERS.labels(service=service).inc()

                    logger.warning(
                        f"🔄 RETRAINING TRIGGERED for {service}: {'; '.join(trigger_reason)}"
                    )

        except Exception as e:
            logger.error(f"Failed to check retraining trigger: {e}", exc_info=True)

    async def _update_satisfaction_metrics(self):
        """Update Prometheus satisfaction metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get overall satisfaction (last 24 hours)
                stats = await conn.fetchrow("""
                    SELECT
                        COUNT(CASE WHEN feedback_type = 'thumbs_up' THEN 1 END) as positive,
                        COUNT(CASE WHEN feedback_type = 'thumbs_down' THEN 1 END) as negative,
                        AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE 0 END) as avg_rating
                    FROM feedback
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                """)

                positive = stats['positive']
                negative = stats['negative']
                total = positive + negative

                if total > 0:
                    satisfaction = positive / total
                    SATISFACTION_SCORE.set(satisfaction)

        except Exception as e:
            logger.error(f"Failed to update satisfaction metrics: {e}", exc_info=True)

    async def get_analytics(
        self,
        service: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get feedback analytics

        Returns comprehensive analytics for continuous improvement
        """
        try:
            async with self.db_pool.acquire() as conn:
                # Build query filter
                service_filter = "AND service = $2" if service else ""
                params = [days] if not service else [days, service]

                # Get overall stats
                stats = await conn.fetchrow(f"""
                    SELECT
                        COUNT(*) as total_feedback,
                        AVG(CASE WHEN rating IS NOT NULL THEN rating ELSE 0 END) as avg_rating,
                        COUNT(CASE WHEN feedback_type = 'thumbs_up' THEN 1 END) as positive,
                        COUNT(CASE WHEN feedback_type = 'thumbs_down' THEN 1 END) as negative
                    FROM feedback
                    WHERE created_at > NOW() - INTERVAL '{days} days'
                    {service_filter}
                """, *params)

                total_feedback = stats['total_feedback']
                avg_rating = float(stats['avg_rating']) if stats['avg_rating'] else 0
                positive = stats['positive']
                negative = stats['negative']

                satisfaction_rate = positive / (positive + negative) if (positive + negative) > 0 else 0

                # Get feedback by type
                feedback_by_type = {}
                type_rows = await conn.fetch(f"""
                    SELECT feedback_type, COUNT(*) as count
                    FROM feedback
                    WHERE created_at > NOW() - INTERVAL '{days} days'
                    {service_filter}
                    GROUP BY feedback_type
                """, *params)

                for row in type_rows:
                    feedback_by_type[row['feedback_type']] = row['count']

                # Get feedback by service
                feedback_by_service = {}
                service_rows = await conn.fetch(f"""
                    SELECT service, COUNT(*) as count
                    FROM feedback
                    WHERE created_at > NOW() - INTERVAL '{days} days'
                    GROUP BY service
                """, days)

                for row in service_rows:
                    feedback_by_service[row['service']] = row['count']

                # Get recent comments
                comment_rows = await conn.fetch(f"""
                    SELECT service, feedback_type, comment, rating, created_at
                    FROM feedback
                    WHERE comment IS NOT NULL
                    AND created_at > NOW() - INTERVAL '{days} days'
                    {service_filter}
                    ORDER BY created_at DESC
                    LIMIT 20
                """, *params)

                recent_comments = [
                    {
                        "service": row['service'],
                        "type": row['feedback_type'],
                        "comment": row['comment'],
                        "rating": row['rating'],
                        "timestamp": row['created_at'].isoformat()
                    }
                    for row in comment_rows
                ]

                # Generate improvement suggestions
                improvement_suggestions = self._generate_improvement_suggestions(
                    satisfaction_rate, avg_rating, feedback_by_type, recent_comments
                )

                # Get retraining recommendations
                retraining_rows = await conn.fetch("""
                    SELECT service, trigger_reason, feedback_count,
                           satisfaction_score, triggered_at, status
                    FROM retraining_triggers
                    WHERE triggered_at > NOW() - INTERVAL '30 days'
                    ORDER BY triggered_at DESC
                    LIMIT 10
                """)

                retraining_recommendations = [
                    {
                        "service": row['service'],
                        "reason": row['trigger_reason'],
                        "feedback_count": row['feedback_count'],
                        "satisfaction": row['satisfaction_score'],
                        "triggered_at": row['triggered_at'].isoformat(),
                        "status": row['status']
                    }
                    for row in retraining_rows
                ]

                return {
                    "total_feedback": total_feedback,
                    "average_rating": round(avg_rating, 2),
                    "satisfaction_rate": round(satisfaction_rate, 3),
                    "feedback_by_type": feedback_by_type,
                    "feedback_by_service": feedback_by_service,
                    "recent_comments": recent_comments,
                    "improvement_suggestions": improvement_suggestions,
                    "retraining_recommendations": retraining_recommendations
                }

        except Exception as e:
            logger.error(f"Failed to get analytics: {e}", exc_info=True)
            raise
    def _generate_improvement_suggestions(
        self,
        satisfaction_rate: float,
        avg_rating: float,
        feedback_by_type: Dict[str, int],
        recent_comments: List[Dict]
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []

        # Satisfaction-based suggestions
        if satisfaction_rate < 0.7:
            suggestions.append(
                f"⚠️ Low satisfaction rate ({satisfaction_rate:.1%}). "
                "Review recent negative feedback and identify common issues."
            )

        # Rating-based suggestions
        if avg_rating > 0 and avg_rating < 3.5:
            suggestions.append(
                f"⚠️ Low average rating ({avg_rating:.1f}/5.0). "
                "Consider retraining models with recent feedback data."
            )

        # Bug reports
        bug_count = feedback_by_type.get('bug_report', 0)
        if bug_count > 5:
            suggestions.append(
                f"🐛 {bug_count} bug reports received. "
                "Prioritize bug fixes before next deployment."
            )

        # Feature requests
        feature_count = feedback_by_type.get('feature_request', 0)
        if feature_count > 10:
            suggestions.append(
                f"💡 {feature_count} feature requests received. "
                "Review and prioritize most requested features."
            )

        # Comment analysis
        if recent_comments:
            # Simple keyword analysis
            common_keywords = {}
            for comment in recent_comments:
                if comment.get('comment'):
                    words = comment['comment'].lower().split()
                    for word in words:
                        if len(word) > 4:  # Skip short words
                            common_keywords[word] = common_keywords.get(word, 0) + 1

            # Find top keywords
            top_keywords = sorted(common_keywords.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_keywords:
                keywords_str = ", ".join([f"{word} ({count})" for word, count in top_keywords])
                suggestions.append(
                    f"🔍 Common feedback keywords: {keywords_str}. "
                    "Review these areas for improvement."
                )

        if not suggestions:
            suggestions.append("✅ No critical issues detected. Continue monitoring feedback.")

        return suggestions


# Initialize service
feedback_service = FeedbackService()


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    try:
        await feedback_service.initialize()
        logger.info("Feedback service started successfully")
    except Exception as e:
        logger.error(f"Failed to start feedback service: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await feedback_service.close()


@app.post("/feedback", response_model=FeedbackResponse)
@ACTIVE_REQUESTS.track_inprogress()
@FEEDBACK_DURATION.time()
async def submit_feedback(request: FeedbackRequest, http_request: Request):
    """
    Submit user feedback

    CRITICAL ENDPOINT: Enables continuous improvement through user feedback

    Example:
    ```json
    {
        "feedback_type": "rating",
        "service": "llm",
        "rating": 4,
        "comment": "Great answer but could be more detailed",
        "query": "How do I recycle plastic bottles?",
        "session_id": "abc123"
    }
    ```
    """
    try:
        result = await feedback_service.submit_feedback(request)
        return FeedbackResponse(**result)
    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@app.get("/analytics", response_model=FeedbackAnalytics)
async def get_analytics(
    service: Optional[str] = None,
    days: int = 7
):
    """
    Get feedback analytics

    Query parameters:
    - service: Filter by service (llm, vision, rag, kg, orchestrator, overall)
    - days: Number of days to analyze (default: 7)

    Returns comprehensive analytics for continuous improvement
    """
    try:
        analytics = await feedback_service.get_analytics(service=service, days=days)
        return FeedbackAnalytics(**analytics)
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "feedback",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("FEEDBACK_SERVICE_PORT", "8007"))

    logger.info(f"Starting Feedback Service on port {port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )


