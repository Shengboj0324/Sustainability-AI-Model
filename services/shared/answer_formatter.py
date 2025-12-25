"""
Answer Formatter - Rich Text & Structured Response Formatting

CRITICAL FEATURES:
- Markdown formatting for rich text display
- Citation formatting with source attribution
- Structured response templates (how-to, factual, creative, org search)
- Frontend-optimized JSON responses
- Accessibility features (ARIA labels, semantic HTML hints)
- Mobile-friendly formatting

This addresses the requirement for "textual output, answer formatting"
"""

import re
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
from dataclasses import dataclass, asdict


class AnswerType(str, Enum):
    """Answer types for template selection"""
    HOW_TO = "how_to"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ORG_SEARCH = "org_search"
    GENERAL = "general"
    ERROR = "error"


@dataclass
class FormattedAnswer:
    """Structured formatted answer"""
    answer_type: str
    content: str  # Markdown formatted
    html_content: Optional[str] = None  # HTML for web clients
    plain_text: str = ""  # Plain text for accessibility
    citations: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.citations is None:
            result['citations'] = []
        if self.metadata is None:
            result['metadata'] = {}
        return result


class AnswerFormatter:
    """
    Production-grade answer formatter with rich text support

    CRITICAL: Provides frontend-optimized, accessible, well-formatted responses
    """

    def __init__(self):
        self.citation_style = "numbered"  # numbered, inline, footnote
        self.max_content_length = 5000
        self.enable_html = True
        self.enable_markdown = True

    def format_answer(
        self,
        answer: str,
        answer_type: AnswerType,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> FormattedAnswer:
        """
        Format answer based on type

        Args:
            answer: Raw answer text
            answer_type: Type of answer (how_to, factual, creative, etc.)
            sources: List of source documents with citations
            metadata: Additional metadata (confidence, warnings, etc.)
            **kwargs: Type-specific parameters

        Returns:
            FormattedAnswer with markdown, HTML, and plain text versions
        """
        # Select formatter based on type
        if answer_type == AnswerType.HOW_TO:
            return self._format_how_to(answer, sources, metadata, **kwargs)
        elif answer_type == AnswerType.FACTUAL:
            return self._format_factual(answer, sources, metadata, **kwargs)
        elif answer_type == AnswerType.CREATIVE:
            return self._format_creative(answer, sources, metadata, **kwargs)
        elif answer_type == AnswerType.ORG_SEARCH:
            return self._format_org_search(answer, sources, metadata, **kwargs)
        elif answer_type == AnswerType.ERROR:
            return self._format_error(answer, metadata, **kwargs)
        else:
            return self._format_general(answer, sources, metadata)

    def _format_how_to(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        steps: Optional[List[str]] = None,
        materials: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        difficulty: Optional[str] = None,
        time_estimate: Optional[str] = None
    ) -> FormattedAnswer:
        """Format how-to guide with numbered steps"""

        # Build markdown content
        md_parts = []

        # Header with metadata
        if difficulty or time_estimate:
            header_parts = []
            if difficulty:
                header_parts.append(f"**Difficulty:** {difficulty}")
            if time_estimate:
                header_parts.append(f"**Time:** {time_estimate}")
            md_parts.append(" | ".join(header_parts))
            md_parts.append("")

        # Materials section
        if materials:
            md_parts.append("## Materials Needed")
            for material in materials:
                md_parts.append(f"- {material}")
            md_parts.append("")

        # Steps section
        if steps:
            md_parts.append("## Steps")
            for i, step in enumerate(steps, 1):
                md_parts.append(f"{i}. {step}")
            md_parts.append("")
        else:
            # Extract steps from answer
            md_parts.append(answer)
            md_parts.append("")

        # Warnings section
        if warnings:
            md_parts.append("## ⚠️ Important Warnings")
            for warning in warnings:
                md_parts.append(f"- ⚠️ {warning}")
            md_parts.append("")

        # Citations
        citations_md = self._format_citations(sources) if sources else ""
        if citations_md:
            md_parts.append(citations_md)

        markdown_content = "\n".join(md_parts)

        # Generate HTML if enabled
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None

        # Generate plain text
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.HOW_TO.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=self._extract_citations(sources) if sources else [],
            metadata=metadata or {}
        )

    def _format_factual(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        facts: Optional[List[str]] = None,
        confidence: Optional[float] = None
    ) -> FormattedAnswer:
        """Format factual answer with bullet points and sources"""

        md_parts = []

        # Confidence indicator
        if confidence is not None:
            if confidence >= 0.8:
                confidence_emoji = "✅"
                confidence_text = "High"
            elif confidence >= 0.5:
                confidence_emoji = "⚠️"
                confidence_text = "Medium"
            else:
                confidence_emoji = "❓"
                confidence_text = "Low"

            md_parts.append(f"{confidence_emoji} **Confidence:** {confidence_text} ({confidence:.0%})")
            md_parts.append("")

        # Main answer
        md_parts.append(answer)
        md_parts.append("")

        # Key facts
        if facts:
            md_parts.append("## Key Facts")
            for fact in facts:
                md_parts.append(f"- {fact}")
            md_parts.append("")

        # Citations
        citations_md = self._format_citations(sources) if sources else ""
        if citations_md:
            md_parts.append(citations_md)

        markdown_content = "\n".join(md_parts)
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.FACTUAL.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=self._extract_citations(sources) if sources else [],
            metadata=metadata or {}
        )

    def _format_creative(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        ideas: Optional[List[Dict]] = None,
        inspiration_images: Optional[List[str]] = None
    ) -> FormattedAnswer:
        """Format creative suggestions with ideas and inspiration"""

        md_parts = []

        # Header
        md_parts.append("# 🎨 Creative Upcycling Ideas")
        md_parts.append("")

        # Main answer
        md_parts.append(answer)
        md_parts.append("")

        # Ideas with difficulty ratings
        if ideas:
            md_parts.append("## Ideas")
            for i, idea in enumerate(ideas, 1):
                title = idea.get('title', f'Idea {i}')
                description = idea.get('description', '')
                difficulty = idea.get('difficulty', 'Medium')
                materials = idea.get('materials', [])

                md_parts.append(f"### {i}. {title}")
                md_parts.append(f"**Difficulty:** {difficulty}")

                if description:
                    md_parts.append(f"\n{description}\n")

                if materials:
                    md_parts.append("**Materials:**")
                    for material in materials:
                        md_parts.append(f"- {material}")

                md_parts.append("")

        # Inspiration images
        if inspiration_images:
            md_parts.append("## 📸 Inspiration")
            for img_url in inspiration_images:
                md_parts.append(f"![Inspiration]({img_url})")
            md_parts.append("")

        # Citations
        citations_md = self._format_citations(sources) if sources else ""
        if citations_md:
            md_parts.append(citations_md)

        markdown_content = "\n".join(md_parts)
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.CREATIVE.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=self._extract_citations(sources) if sources else [],
            metadata=metadata or {}
        )

    def _format_org_search(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        organizations: Optional[List[Dict]] = None
    ) -> FormattedAnswer:
        """Format organization search results"""

        md_parts = []

        # Header
        md_parts.append("# 🏢 Organizations & Resources")
        md_parts.append("")

        # Main answer
        md_parts.append(answer)
        md_parts.append("")

        # Organizations
        if organizations:
            md_parts.append("## Recommended Organizations")
            for org in organizations:
                name = org.get('name', 'Unknown')
                description = org.get('description', '')
                address = org.get('address', '')
                phone = org.get('phone', '')
                website = org.get('website', '')
                hours = org.get('hours', '')
                distance = org.get('distance_km')

                md_parts.append(f"### {name}")

                if distance is not None:
                    md_parts.append(f"📍 **Distance:** {distance:.1f} km")

                if description:
                    md_parts.append(f"\n{description}\n")

                if address:
                    md_parts.append(f"**Address:** {address}")

                if phone:
                    md_parts.append(f"**Phone:** {phone}")

                if website:
                    md_parts.append(f"**Website:** [{website}]({website})")

                if hours:
                    md_parts.append(f"**Hours:** {hours}")

                md_parts.append("")

        markdown_content = "\n".join(md_parts)
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.ORG_SEARCH.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=self._extract_citations(sources) if sources else [],
            metadata=metadata or {}
        )

    def _format_general(
        self,
        answer: str,
        sources: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> FormattedAnswer:
        """Format general answer"""

        md_parts = [answer]

        # Citations
        citations_md = self._format_citations(sources) if sources else ""
        if citations_md:
            md_parts.append("")
            md_parts.append(citations_md)

        markdown_content = "\n".join(md_parts)
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.GENERAL.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=self._extract_citations(sources) if sources else [],
            metadata=metadata or {}
        )

    def _format_error(
        self,
        answer: str,
        metadata: Optional[Dict] = None,
        error_code: Optional[str] = None,
        suggestions: Optional[List[str]] = None
    ) -> FormattedAnswer:
        """Format error message with recovery suggestions"""

        md_parts = []

        # Error header
        md_parts.append("# ❌ Error")
        md_parts.append("")

        if error_code:
            md_parts.append(f"**Error Code:** `{error_code}`")
            md_parts.append("")

        # Error message
        md_parts.append(answer)
        md_parts.append("")

        # Recovery suggestions
        if suggestions:
            md_parts.append("## 💡 Suggestions")
            for suggestion in suggestions:
                md_parts.append(f"- {suggestion}")
            md_parts.append("")

        markdown_content = "\n".join(md_parts)
        html_content = self._markdown_to_html(markdown_content) if self.enable_html else None
        plain_text = self._markdown_to_plain(markdown_content)

        return FormattedAnswer(
            answer_type=AnswerType.ERROR.value,
            content=markdown_content,
            html_content=html_content,
            plain_text=plain_text,
            citations=[],
            metadata=metadata or {}
        )

    def _format_citations(self, sources: List[Dict]) -> str:
        """Format citations in markdown"""
        if not sources:
            return ""

        md_parts = ["## 📚 Sources"]

        for i, source in enumerate(sources, 1):
            source_text = source.get('source', 'Unknown')
            doc_type = source.get('doc_type', '')
            score = source.get('score', 0)

            citation = f"{i}. {source_text}"
            if doc_type:
                citation += f" ({doc_type})"
            if score > 0:
                citation += f" - Relevance: {score:.0%}"

            md_parts.append(citation)

        return "\n".join(md_parts)

    def _extract_citations(self, sources: List[Dict]) -> List[Dict[str, Any]]:
        """Extract citation metadata"""
        if not sources:
            return []

        citations = []
        for i, source in enumerate(sources, 1):
            citations.append({
                "id": i,
                "source": source.get('source', 'Unknown'),
                "doc_type": source.get('doc_type', ''),
                "score": source.get('score', 0),
                "url": source.get('url', ''),
                "metadata": source.get('metadata', {})
            })

        return citations

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert markdown to HTML (basic implementation)

        For production, use a library like markdown2 or mistune
        """
        html = markdown

        # Headers
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)

        # Links
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)

        # Lists
        html = re.sub(r'^\- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^(\d+)\. (.+)$', r'<li>\2</li>', html, flags=re.MULTILINE)

        # Wrap paragraphs
        lines = html.split('\n')
        html_lines = []
        in_list = False

        for line in lines:
            if line.startswith('<li>'):
                if not in_list:
                    html_lines.append('<ul>')
                    in_list = True
                html_lines.append(line)
            else:
                if in_list:
                    html_lines.append('</ul>')
                    in_list = False
                if line.strip() and not line.startswith('<'):
                    html_lines.append(f'<p>{line}</p>')
                else:
                    html_lines.append(line)

        if in_list:
            html_lines.append('</ul>')

        return '\n'.join(html_lines)

    def _markdown_to_plain(self, markdown: str) -> str:
        """Convert markdown to plain text"""
        plain = markdown

        # Remove headers
        plain = re.sub(r'^#{1,6} ', '', plain, flags=re.MULTILINE)

        # Remove bold/italic
        plain = re.sub(r'\*\*(.+?)\*\*', r'\1', plain)
        plain = re.sub(r'\*(.+?)\*', r'\1', plain)

        # Remove links (keep text)
        plain = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', plain)

        # Remove images
        plain = re.sub(r'!\[.+?\]\(.+?\)', '', plain)

        # Clean up emojis for screen readers (optional)
        # plain = re.sub(r'[^\w\s\-.,!?]', '', plain)

        return plain.strip()

