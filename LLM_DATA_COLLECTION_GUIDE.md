# LLM Training Data Collection Guide

## 🎯 **OBJECTIVE**

Collect **1,000,000+ high-quality, creative upcycling examples** for training Llama-3-8B with LoRA fine-tuning.

**Target Breakdown:**
- **Reddit**: 200,000 examples (real community discussions)
- **YouTube**: 100,000 examples (tutorial transcripts)
- **Synthetic (GPT-4)**: 700,000 examples (creative, diverse, innovative)

---

## 📋 **PREREQUISITES**

### 1. Install Dependencies

```bash
pip install -r requirements_llm_data_collection.txt
```

### 2. API Credentials Setup

#### **Reddit API** (Free)
1. Go to: https://www.reddit.com/prefs/apps
2. Click "Create App" → Select "script"
3. Get `client_id` and `client_secret`
4. Set environment variables:
```bash
export REDDIT_CLIENT_ID="your_client_id"
export REDDIT_CLIENT_SECRET="your_client_secret"
```

#### **YouTube API** (Free - 10K quota/day)
1. Go to: https://console.cloud.google.com/apis/credentials
2. Create project → Enable YouTube Data API v3
3. Create API key
4. Set environment variable:
```bash
export YOUTUBE_API_KEY="your_api_key"
```

#### **OpenAI API** (Paid - ~$28K for 700K examples)
1. Go to: https://platform.openai.com/api-keys
2. Create API key
3. Add credits to account
4. Set environment variable:
```bash
export OPENAI_API_KEY="your_api_key"
```

**Cost Estimate:**
- GPT-4 Turbo: ~$0.04 per example
- 700,000 examples × $0.04 = **$28,000**
- Budget alternative: Generate 100K examples for **$4,000**

---

## 🚀 **USAGE**

### **Option 1: Run Complete Pipeline (Recommended)**

```bash
cd scripts/data
python3 collect_llm_training_data.py
```

This will:
1. Scrape Reddit (200K examples)
2. Scrape YouTube (100K examples)
3. Generate synthetic data (700K examples)
4. Quality control & deduplication
5. Train/val split (95/5)
6. Save to `data/processed/llm_sft/`

**Estimated Time:**
- Reddit: 6-8 hours
- YouTube: 4-6 hours (quota limited)
- Synthetic: 48-72 hours (API rate limits)
- **Total: 3-4 days**

---

### **Option 2: Run Individual Scrapers**

#### **Reddit Only**
```bash
python3 scripts/data/scrape_reddit_upcycling.py
```
- Output: `data/raw/llm/reddit/reddit_upcycling_raw.jsonl`
- Time: 6-8 hours
- Cost: Free

#### **YouTube Only**
```bash
python3 scripts/data/scrape_youtube_tutorials.py
```
- Output: `data/raw/llm/youtube/youtube_tutorials_raw.jsonl`
- Time: 4-6 hours (quota: 10K/day)
- Cost: Free

#### **Synthetic Only**
```bash
python3 scripts/data/generate_synthetic_creative.py
```
- Output: `data/raw/llm/synthetic/synthetic_creative.jsonl`
- Time: 48-72 hours
- Cost: ~$28,000 (700K examples) or ~$4,000 (100K examples)

**To reduce synthetic count (budget mode):**
Edit `scripts/data/generate_synthetic_creative.py`:
```python
TARGET_COUNT = 100000  # Instead of 500000
```

---

## 📊 **OUTPUT FORMAT**

All data follows OpenAI chat format:

```json
{
  "messages": [
    {"role": "user", "content": "How can I upcycle plastic bottles?"},
    {"role": "assistant", "content": "Here are 5 creative ways to upcycle plastic bottles:\n\n1. **Vertical Garden**: Cut bottles in half, fill with soil...\n2. **Bird Feeder**: Cut openings, add perches...\n..."}
  ],
  "category": "upcycling_ideas",
  "metadata": {
    "source": "reddit",
    "subreddit": "upcycling",
    "score": 245,
    "creativity_score": 0.78,
    "scraped_at": "2024-11-20T10:30:00"
  }
}
```

---

## 🔍 **QUALITY CONTROL**

### **Automated Validation**
- ✅ Deduplication (MD5 content hashing)
- ✅ Length validation (30-2000 words)
- ✅ Creativity scoring (keyword analysis)
- ✅ Safety checks (harmful content filtering)
- ✅ Spam detection (banned keywords)

### **Quality Metrics**
- **Reddit**: Minimum post score, creativity score ≥ 0.3
- **YouTube**: Manual transcripts preferred, view count ≥ 1K
- **Synthetic**: GPT-4 temperature 0.9 for creativity

---

## 📈 **MONITORING PROGRESS**

### **Check Statistics**
```bash
# Reddit stats
cat data/raw/llm/reddit/reddit_scraping_stats.json

# YouTube stats
cat data/raw/llm/youtube/youtube_scraping_stats.json

# Synthetic stats
cat data/raw/llm/synthetic/generation_stats.json

# Final collection stats
cat data/processed/llm_sft/collection_stats.json
```

### **Count Examples**
```bash
wc -l data/processed/llm_sft/sustainability_creative_train.jsonl
wc -l data/processed/llm_sft/sustainability_creative_val.jsonl
```

---

## ⚠️ **TROUBLESHOOTING**

### **Reddit API Errors**
- **429 Rate Limit**: Increase `REQUEST_DELAY` in script
- **401 Unauthorized**: Check credentials
- **403 Forbidden**: User agent blocked, change `REDDIT_USER_AGENT`

### **YouTube API Errors**
- **Quota Exceeded**: Wait 24 hours or create new project
- **No Transcript**: Video has no captions (expected, will skip)
- **403 Forbidden**: API key invalid or API not enabled

### **OpenAI API Errors**
- **429 Rate Limit**: Reduce `BATCH_SIZE` or add delays
- **Insufficient Credits**: Add funds to account
- **Invalid API Key**: Check `OPENAI_API_KEY` environment variable

---

## 💰 **COST OPTIMIZATION**

### **Budget Mode (Total: ~$40)**
1. **Reddit**: 200K examples (Free)
2. **YouTube**: 100K examples (Free)
3. **Synthetic**: 100K examples ($4,000)
4. **Total**: 400K examples

Edit `collect_llm_training_data.py`:
```python
self.phase_3_synthetic(target=100000)  # Instead of 700000
```

### **Free Mode (Total: $0)**
1. **Reddit**: 200K examples
2. **YouTube**: 100K examples
3. **Skip synthetic generation**
4. **Total**: 300K examples

Comment out in `collect_llm_training_data.py`:
```python
# self.phase_3_synthetic(target=700000)
```

---

## ✅ **VERIFICATION**

After collection completes, verify data quality:

```bash
# Check file sizes
ls -lh data/processed/llm_sft/

# Sample random examples
head -5 data/processed/llm_sft/sustainability_creative_train.jsonl | python3 -m json.tool

# Validate JSON format
python3 -c "
import json
with open('data/processed/llm_sft/sustainability_creative_train.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Invalid JSON at line {i+1}')
            break
    else:
        print('✅ All JSON valid')
"
```

---

## 🎓 **NEXT STEPS**

After data collection:

1. **Update training config** for RTX 5090
2. **Increase LoRA rank** to 256 (from 64)
3. **Update batch size** for 24GB VRAM
4. **Start training** with 1M examples

See: `configs/llm_sft_rtx5090.yaml` (to be created)


