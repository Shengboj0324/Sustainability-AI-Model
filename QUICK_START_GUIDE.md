# QUICK START GUIDE - LLM Data Collection
## ReleAF AI - Phase 1 & 2

**Status**: ✅ PRODUCTION READY  
**Last Updated**: 2024-11-20

---

## 🚀 QUICK START (5 MINUTES)

### **Step 1: Install Dependencies**
```bash
pip install -r requirements_llm_data_collection.txt
```

### **Step 2: Set Up API Credentials**

#### **Reddit API** (Free - Required for Phase 1)
```bash
# Get credentials at: https://www.reddit.com/prefs/apps
export REDDIT_CLIENT_ID="your_client_id_here"
export REDDIT_CLIENT_SECRET="your_client_secret_here"
```

#### **YouTube API** (Free - Required for Phase 1)
```bash
# Get API key at: https://console.cloud.google.com/apis/credentials
export YOUTUBE_API_KEY="your_api_key_here"
```

#### **OpenAI API** (Paid - Optional for Phase 2)
```bash
# Get API key at: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your_api_key_here"
```

### **Step 3: Run Data Collection**

#### **Option A: Free Mode (300K examples, $0)**
```bash
# Edit collect_llm_training_data.py and comment out phase_3_synthetic
python3 scripts/data/collect_llm_training_data.py
```

#### **Option B: Budget Mode (400K examples, $4K)**
```bash
# Edit generate_synthetic_creative.py: TARGET_COUNT = 100000
python3 scripts/data/collect_llm_training_data.py
```

#### **Option C: Full Mode (1M examples, $28K)**
```bash
# Use default settings
python3 scripts/data/collect_llm_training_data.py
```

### **Step 4: Verify Data**
```bash
# Check file sizes
wc -l data/processed/llm_sft/*.jsonl

# Inspect samples
head -5 data/processed/llm_sft/sustainability_creative_train.jsonl | python3 -m json.tool
```

### **Step 5: Start Training**
```bash
# RTX 5090
python3 training/llm/train_sft.py --config configs/llm_sft_rtx5090.yaml

# M4 Max
python3 training/llm/train_sft.py --config configs/llm_sft_m4max.yaml
```

---

## 📋 DETAILED INSTRUCTIONS

### **Reddit API Setup**

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Fill in:
   - **Name**: ReleAF-AI-Data-Collector
   - **Type**: Script
   - **Description**: Data collection for sustainability AI
   - **Redirect URI**: http://localhost:8080
4. Click "Create app"
5. Copy **client_id** (under app name) and **secret**
6. Set environment variables:
   ```bash
   export REDDIT_CLIENT_ID="your_14_char_id"
   export REDDIT_CLIENT_SECRET="your_27_char_secret"
   ```

### **YouTube API Setup**

1. Go to https://console.cloud.google.com/
2. Create a new project or select existing
3. Enable "YouTube Data API v3"
4. Go to "Credentials" → "Create Credentials" → "API Key"
5. Copy the API key
6. Set environment variable:
   ```bash
   export YOUTUBE_API_KEY="your_39_char_key"
   ```

### **OpenAI API Setup** (Optional)

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Set environment variable:
   ```bash
   export OPENAI_API_KEY="sk-your_key_here"
   ```

---

## 🎯 RUNNING INDIVIDUAL SCRAPERS

### **Reddit Only**
```bash
python3 scripts/data/scrape_reddit_upcycling.py
```
- **Output**: `data/raw/llm/reddit/reddit_upcycling_raw.jsonl`
- **Time**: 6-8 hours
- **Target**: 200K examples
- **Cost**: FREE

### **YouTube Only**
```bash
python3 scripts/data/scrape_youtube_tutorials.py
```
- **Output**: `data/raw/llm/youtube/youtube_tutorials_raw.jsonl`
- **Time**: 4-6 hours (quota limited)
- **Target**: 100K examples
- **Cost**: FREE

### **Synthetic Only**
```bash
python3 scripts/data/generate_synthetic_creative.py
```
- **Output**: `data/raw/llm/synthetic/synthetic_creative.jsonl`
- **Time**: 48-72 hours
- **Target**: 700K examples (configurable)
- **Cost**: ~$28,000 (or $4,000 for 100K)

---

## 🔧 CONFIGURATION

### **Adjust Target Counts**

Edit the respective files:

**Reddit** (`scrape_reddit_upcycling.py`):
```python
SUBREDDITS = {
    "upcycling": {"priority": "CRITICAL", "target": 50000, ...},
    # Adjust target values
}
```

**YouTube** (`scrape_youtube_tutorials.py`):
```python
def run(self, target_count: int = 100000):  # Change this
```

**Synthetic** (`generate_synthetic_creative.py`):
```python
TARGET_COUNT = 500000  # Change this (default: 500K)
```

### **Adjust Rate Limits**

**Reddit**:
```python
REQUESTS_PER_MINUTE = 55  # Max 60, keep conservative
```

**YouTube**:
```python
DAILY_QUOTA = 10000  # YouTube API limit
```

**Synthetic**:
```python
time.sleep(0.5)  # Delay between batches (line 352)
```

---

## 📊 MONITORING PROGRESS

### **Check Logs**
All scrapers log to console with timestamps:
```
2024-11-20 14:30:15 - INFO - ✅ Collected 1000 examples
2024-11-20 14:30:20 - INFO - Checkpoint saved
```

### **Check Checkpoints**
Scrapers save checkpoints periodically:
```bash
# Reddit: every 100 posts
ls -lh data/raw/llm/reddit/reddit_checkpoint.jsonl

# YouTube: every 50 videos
ls -lh data/raw/llm/youtube/youtube_checkpoint.jsonl

# Synthetic: every 100 batches
ls -lh data/raw/llm/synthetic/synthetic_creative_checkpoint.jsonl
```

### **Check Statistics**
Each scraper saves statistics:
```bash
cat data/raw/llm/reddit/reddit_scraping_stats.json
cat data/raw/llm/youtube/youtube_scraping_stats.json
cat data/raw/llm/synthetic/generation_stats.json
```

---

## 🛠️ TROUBLESHOOTING

### **Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/Sustainability-AI-Model

# Install dependencies
pip install -r requirements_llm_data_collection.txt
```

### **API Credential Errors**
```bash
# Verify environment variables are set
echo $REDDIT_CLIENT_ID
echo $REDDIT_CLIENT_SECRET
echo $YOUTUBE_API_KEY
echo $OPENAI_API_KEY

# Re-export if needed
export REDDIT_CLIENT_ID="..."
```

### **Rate Limit Errors**
- **Reddit**: Wait 1 minute, script will auto-retry
- **YouTube**: Quota resets daily at midnight PT
- **OpenAI**: Script has exponential backoff (2, 4, 8 seconds)

### **Crash Recovery**
If a scraper crashes, just re-run it:
```bash
# It will automatically load the checkpoint and resume
python3 scripts/data/scrape_reddit_upcycling.py
```

### **Memory Issues**
If you see OOM errors:
```bash
# Reduce batch sizes in orchestrator
# Or run scrapers individually instead of all at once
```

---

## ✅ VERIFICATION

### **Test Installation**
```bash
python3 scripts/data/test_data_collection.py
```
Expected output:
```
🎉 ALL TESTS PASSED! 🎉
Total: 4/4 tests passed
```

### **Verify Data Quality**
```bash
# Check line counts
wc -l data/processed/llm_sft/*.jsonl

# Inspect random samples
shuf -n 5 data/processed/llm_sft/sustainability_creative_train.jsonl | python3 -m json.tool

# Check for duplicates
cat data/processed/llm_sft/sustainability_creative_train.jsonl | \
  python3 -c "import sys, json, hashlib; \
  hashes = set(); \
  for line in sys.stdin: \
    item = json.loads(line); \
    h = hashlib.sha256(str(item['messages']).encode()).hexdigest(); \
    if h in hashes: print('DUPLICATE FOUND'); \
    hashes.add(h); \
  print(f'Total unique: {len(hashes)}')"
```

---

## 📈 EXPECTED TIMELINE

| Phase | Time | Output |
|-------|------|--------|
| Reddit scraping | 6-8h | 200K examples |
| YouTube scraping | 4-6h | 100K examples |
| Synthetic (100K) | 8-12h | 100K examples |
| Synthetic (700K) | 48-72h | 700K examples |
| Quality control | 1-2h | Deduplication |
| **Total (Free)** | **12-14h** | **300K** |
| **Total (Budget)** | **1 day** | **400K** |
| **Total (Full)** | **3-4 days** | **1M** |

---

## 💰 COST BREAKDOWN

| Option | Reddit | YouTube | Synthetic | Total Cost | Total Examples |
|--------|--------|---------|-----------|------------|----------------|
| **Free** | $0 | $0 | $0 | **$0** | 300K |
| **Budget** | $0 | $0 | $4,000 | **$4,000** | 400K |
| **Full** | $0 | $0 | $28,000 | **$28,000** | 1M |

---

## 🎓 NEXT STEPS

After data collection:

1. **Verify data quality** (see Verification section)
2. **Update training config** (`configs/llm_sft_rtx5090.yaml`)
3. **Run training** on RTX 5090 or M4 Max
4. **Evaluate model** on validation set
5. **Deploy** to production

---

**For detailed documentation, see**:
- `LLM_DATA_COLLECTION_GUIDE.md` - Complete user guide
- `FINAL_IMPLEMENTATION_REPORT.md` - Technical details
- `INTENSIVE_CODE_QUALITY_AUDIT.md` - Quality audit results

