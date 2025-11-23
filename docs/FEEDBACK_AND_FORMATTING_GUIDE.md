# 📚 Feedback & Answer Formatting Guide
## ReleAF AI - User Feedback and Rich Text Formatting

---

## 🎯 Overview

This guide covers the new features for:
1. **User Feedback Collection** - Continuous improvement through user input
2. **Rich Answer Formatting** - Markdown, HTML, and accessible text output
3. **Frontend Integration** - Structured responses for UI rendering

---

## 1. User Feedback System

### Starting the Feedback Service

```bash
# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=releaf_feedback
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your_password
export FEEDBACK_SERVICE_PORT=8007

# Start the service
cd services/feedback_service
python3 server.py
```

### Submitting Feedback

**Endpoint**: `POST http://localhost:8007/feedback`

**Example 1: Thumbs Up**
```json
{
  "feedback_type": "thumbs_up",
  "service": "llm",
  "query": "How do I recycle plastic bottles?",
  "response": "Plastic bottles can be recycled...",
  "session_id": "abc123",
  "user_id": "user_456"
}
```

**Example 2: Rating with Comment**
```json
{
  "feedback_type": "rating",
  "service": "orchestrator",
  "rating": 4,
  "comment": "Great answer but could be more detailed",
  "query": "What can I make from old t-shirts?",
  "response": "You can make tote bags, rugs...",
  "session_id": "abc123",
  "metadata": {"confidence": 0.85}
}
```

**Example 3: Bug Report**
```json
{
  "feedback_type": "bug_report",
  "service": "vision",
  "comment": "Image recognition failed for clear plastic",
  "query": "What type of plastic is this?",
  "session_id": "abc123"
}
```

### Getting Analytics

**Endpoint**: `GET http://localhost:8007/analytics?service=llm&days=7`

**Response**:
```json
{
  "total_feedback": 150,
  "average_rating": 4.2,
  "satisfaction_rate": 0.85,
  "feedback_by_type": {
    "thumbs_up": 80,
    "thumbs_down": 15,
    "rating": 45,
    "comment": 10
  },
  "feedback_by_service": {
    "llm": 60,
    "vision": 40,
    "rag": 30,
    "orchestrator": 20
  },
  "recent_comments": [
    {
      "service": "llm",
      "type": "rating",
      "comment": "Very helpful!",
      "rating": 5,
      "timestamp": "2025-11-22T10:30:00"
    }
  ],
  "improvement_suggestions": [
    "✅ No critical issues detected. Continue monitoring feedback."
  ],
  "retraining_recommendations": []
}
```

---

## 2. Rich Answer Formatting

### Using the Answer Formatter

```python
from services.shared.answer_formatter import AnswerFormatter, AnswerType

formatter = AnswerFormatter()

# Format a how-to guide
formatted = formatter.format_answer(
    answer="Here's how to upcycle a plastic bottle into a planter.",
    answer_type=AnswerType.HOW_TO,
    steps=[
        "Cut the bottle in half",
        "Drill drainage holes in the bottom",
        "Add soil and plant seeds"
    ],
    materials=["Plastic bottle", "Scissors", "Drill", "Soil", "Seeds"],
    warnings=["Use caution when cutting plastic"],
    difficulty="Easy",
    time_estimate="15 minutes"
)

# Access different formats
print(formatted.content)       # Markdown
print(formatted.html_content)  # HTML
print(formatted.plain_text)    # Plain text (accessibility)
print(formatted.citations)     # Structured citations
```

### Answer Types

**1. HOW_TO** - Step-by-step guides
```python
formatted = formatter.format_answer(
    answer="Transform your old t-shirt!",
    answer_type=AnswerType.HOW_TO,
    steps=["Step 1", "Step 2", "Step 3"],
    materials=["Material 1", "Material 2"],
    warnings=["Warning 1"],
    difficulty="Medium",
    time_estimate="30 minutes"
)
```

**2. FACTUAL** - Factual answers with confidence
```python
formatted = formatter.format_answer(
    answer="Plastic bottles are recyclable.",
    answer_type=AnswerType.FACTUAL,
    sources=[{"source": "EPA Guidelines", "score": 0.95}],
    confidence=0.92,
    facts=["Fact 1", "Fact 2", "Fact 3"]
)
```

**3. CREATIVE** - Creative upcycling ideas
```python
formatted = formatter.format_answer(
    answer="Transform your old items!",
    answer_type=AnswerType.CREATIVE,
    ideas=[
        {
            "title": "Tote Bag",
            "description": "Make a reusable bag",
            "difficulty": "Easy",
            "materials": ["Old t-shirt", "Scissors"]
        }
    ]
)
```

**4. ORG_SEARCH** - Organization listings
```python
formatted = formatter.format_answer(
    answer="Here are recycling centers near you:",
    answer_type=AnswerType.ORG_SEARCH,
    organizations=[
        {
            "name": "Green Recycling Center",
            "address": "123 Main St",
            "phone": "(555) 123-4567",
            "website": "https://example.com",
            "hours": "Mon-Fri 8AM-6PM",
            "distance_km": 2.3
        }
    ]
)
```

**5. ERROR** - Error messages with suggestions
```python
formatted = formatter.format_answer(
    answer="Unable to process your request.",
    answer_type=AnswerType.ERROR,
    error_code="LOW_IMAGE_QUALITY",
    suggestions=[
        "Try taking a photo in better lighting",
        "Move closer to the object"
    ]
)
```

---

## 3. Frontend Integration

### Orchestrator Response with Rich Formatting

**Endpoint**: `POST http://localhost:8001/orchestrate`

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": "How do I recycle plastic bottles?"}
  ]
}
```

**Response** (with new fields):
```json
{
  "response": "Plastic bottles can be recycled...",
  "confidence_score": 0.85,
  "confidence_level": "high",
  "sources": [
    {"source": "EPA Guidelines", "doc_type": "official"}
  ],
  "formatted_answer": {
    "answer_type": "factual",
    "content": "✅ **Confidence:** High (85%)\n\nPlastic bottles...",
    "html_content": "<p><strong>Confidence:</strong> High (85%)</p>...",
    "plain_text": "Confidence: High (85%) Plastic bottles...",
    "citations": [
      {
        "id": 1,
        "source": "EPA Guidelines",
        "doc_type": "official",
        "score": 0.95
      }
    ],
    "metadata": {"confidence": 0.85}
  },
  "answer_type": "factual",
  "citations": [...],
  "response_id": "a1b2c3d4e5f6g7h8",
  "processing_time_ms": 1250.5
}
```

### Frontend Rendering Example (React)

```jsx
function AnswerDisplay({ response }) {
  // Choose format based on client capabilities
  const content = response.formatted_answer.html_content || 
                  response.formatted_answer.content;
  
  return (
    <div className="answer">
      {/* Render HTML or Markdown */}
      <div dangerouslySetInnerHTML={{ __html: content }} />
      
      {/* Render citations */}
      {response.citations && (
        <div className="citations">
          <h3>Sources</h3>
          {response.citations.map(citation => (
            <div key={citation.id}>
              {citation.id}. {citation.source}
              {citation.url && (
                <a href={citation.url} target="_blank">View Source</a>
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* Feedback buttons */}
      <div className="feedback">
        <button onClick={() => submitFeedback('thumbs_up', response.response_id)}>
          👍
        </button>
        <button onClick={() => submitFeedback('thumbs_down', response.response_id)}>
          👎
        </button>
      </div>
    </div>
  );
}

async function submitFeedback(type, responseId) {
  await fetch('http://localhost:8007/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      feedback_type: type,
      service: 'orchestrator',
      session_id: sessionStorage.getItem('session_id'),
      metadata: { response_id: responseId }
    })
  });
}
```

---

## 4. Continuous Improvement Workflow

### Automated Retraining Triggers

The system automatically triggers retraining when:
- **100+ feedback** received in 7 days
- **Satisfaction rate < 60%** (thumbs down / total)
- **20+ negative feedback** reports
- **Average rating < 3.0/5.0**

### Monitoring Retraining Triggers

```bash
# Check Prometheus metrics
curl http://localhost:8007/metrics | grep retraining_triggers

# Check database
psql -d releaf_feedback -c "SELECT * FROM retraining_triggers ORDER BY triggered_at DESC LIMIT 10;"
```

---

## 5. Best Practices

### For Frontend Developers
1. **Always use `formatted_answer`** for rich content
2. **Provide multiple format options** (HTML for web, plain text for accessibility)
3. **Render citations** as clickable links
4. **Collect feedback** after every response
5. **Use `response_id`** to track feedback

### For Backend Developers
1. **Set answer_type** based on task classification
2. **Include sources** for citation generation
3. **Add confidence scores** for transparency
4. **Log all feedback** for continuous improvement
5. **Monitor retraining triggers** regularly

---

## 6. Troubleshooting

### Feedback Service Won't Start
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Create database
createdb releaf_feedback

# Check logs
tail -f logs/feedback_service.log
```

### Formatting Not Working
```python
# Verify answer formatter import
from services.shared.answer_formatter import AnswerFormatter

# Check answer type
print(AnswerType.HOW_TO.value)  # Should print "how_to"
```

---

## 7. API Reference

### Feedback Service Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/feedback` | POST | Submit user feedback |
| `/analytics` | GET | Get feedback analytics |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### Feedback Types

- `thumbs_up` - Positive feedback
- `thumbs_down` - Negative feedback
- `rating` - 1-5 star rating (requires `rating` field)
- `comment` - Text comment
- `bug_report` - Bug report
- `feature_request` - Feature suggestion

### Service Types

- `llm` - Language model
- `vision` - Vision service
- `rag` - RAG service
- `kg` - Knowledge graph
- `orchestrator` - Orchestrator
- `overall` - Overall system

---

**For more information, see**:
- `DEEP_ANALYSIS_IMPLEMENTATION_REPORT.md` - Full implementation details
- `tests/test_deep_integration.py` - Usage examples
- `services/feedback_service/server.py` - Feedback service code
- `services/shared/answer_formatter.py` - Answer formatter code

