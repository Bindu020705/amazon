# AWS Jupyter Notebook Complete Guide: Business Entity Resolution Challenge

**Target Score:** 99th percentile (F₀.₅ ≥ 0.85+)
**Current Baseline:** F₀.₅ = 0.772 (holdout)

---

## 📋 TABLE OF CONTENTS

1. [AWS Environment Setup](#1-aws-environment-setup)
2. [Data Upload & Exploration](#2-data-upload--exploration)
3. [Pipeline Architecture Deep Dive](#3-pipeline-architecture-deep-dive)
4. [Running the Baseline End-to-End](#4-running-the-baseline-end-to-end)
5. [Validation & Submission](#5-validation--submission)
6. [Advanced Tuning for 99th Percentile](#6-advanced-tuning-for-99th-percentile)
7. [Troubleshooting & Monitoring](#7-troubleshooting--monitoring)

---

## 1. AWS ENVIRONMENT SETUP

### 1.1 Recommended AWS Configuration

| Component | Specification | Rationale |
|-----------|---------------|-----------|
| **Instance Type** | `r6g.4xlarge` (16 vCPU, 128 GB RAM) or `r6i.4xlarge` | Memory-mapped stores need RAM; 128 GB handles full test corpus |
| **Storage** | 500 GB gp3 EBS (3000 IOPS) | Dataset ~2.4 GB + working files ~10 GB |
| **OS** | Ubuntu 22.04 / Amazon Linux 2023 | Python 3.10+ support |
| **Network** | VPC with S3 access | For model artifact upload/download |

### 1.2 Launch Options

#### Option A: SageMaker Notebook Instance (Easiest)
```bash
# In AWS Console:
# 1. SageMaker → Notebook instances → Create notebook instance
# 2. Instance type: ml.r6g.4xlarge (or ml.r5.4xlarge)
# 3. Volume size: 500 GB
# 4. IAM Role: SageMakerFullAccess + S3ReadWrite
# 5. Open Jupyter → New → Terminal
```

#### Option B: EC2 + Jupyter (More Control)
```bash
# Launch EC2 (r6g.4xlarge, 500 GB gp3)
# SSH in:
sudo apt update && sudo apt install -y python3.11 python3.11-venv git htop nvtop
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# Tunnel: ssh -L 8888:localhost:8888 ubuntu@<EC2_IP>
```

#### Option C: SageMaker Studio (Best for Teams)
- Shared environment, persistent storage, built-in Git

---

### 1.3 Clone & Setup Project

```bash
# In Jupyter Terminal
cd /home/ubuntu  # or /home/ec2-user
git clone <your-repo-url> business-entity-resolution
cd business-entity-resolution/student_resource

# Verify structure
tree -L 3
# Should see: dataset/, code/, utils/, output/, work/, Documentation_template.md
```

### 1.4 Install Dependencies

```bash
# In Jupyter Terminal
cd code/business_entity_resolution
pip install -r requirements.txt
# Verify:
python -c "import lightgbm, numba, polars, rapidfuzz, numpy; print('OK')"
```

**requirements.txt:**
```txt
numpy
lightgbm
numba
polars
rapidfuzz
```

---

## 2. DATA UPLOAD & EXPLORATION

### 2.1 Data Layout (Already Present)

```
dataset/
├── train/
│   ├── train_source1.tsv      # 210 MB  (S1 reference)
│   ├── train_source2.tsv      # 489 MB
│   ├── train_source3.tsv      # 504 MB
│   └── train_ground_truth.tsv # 127 MB
└── test/
    ├── test_source1.tsv       # 175 MB
    ├── test_source2.tsv       # 509 MB
    └── test_source3.tsv       # 506 MB
```

### 2.2 Quick EDA Notebook

Create `eda.ipynb` in `student_resource/`:

```python
# %% [markdown]
# # Exploratory Data Analysis

# %%
import polars as pl
import numpy as np

# Load sample
train_s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=10000)
train_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=10000)
train_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=10000)
gt = pl.read_csv("dataset/train/train_ground_truth.tsv", separator="\t")

print("S1 shape:", train_s1.shape)
print("S2 shape:", train_s2.shape)
print("S3 shape:", train_s3.shape)
print("GT shape:", gt.shape)

# %%
# Country distribution
for df, name in [(train_s1, "S1"), (train_s2, "S2"), (train_s3, "S3")]:
    print(f"\n{name} countries:")
    print(df["country"].value_counts())

# %%
# Ground truth analysis
gt_parsed = gt.with_columns(
    pl.col("matched_entity_ids").str.split(",").alias("matches")
).with_columns(
    pl.col("matches").list.lengths().alias("n_matches")
)
print("\nGT match counts:")
print(gt_parsed["n_matches"].value_counts().sort("n_matches"))

# %%
# Name length analysis
for col in ["business_name", "business_address"]:
    for df, name in [(train_s1, "S1"), (train_s2, "S2"), (train_s3, "S3")]:
        lengths = df[col].str.lengths()
        print(f"{name} {col}: mean={lengths.mean():.1f}, median={lengths.median():.1f}, max={lengths.max()}")

# %%
# Sample records with Indic scripts
indic_s2 = train_s2.filter(pl.col("business_name").str.contains(r"[\u0900-\u0D7F]"))
indic_s3 = train_s3.filter(pl.col("business_name").str.contains(r"[\u0900-\u0D7F]"))
print(f"\nIndic script records: S2={indic_s2.height}, S3={indic_s3.height}")
if indic_s2.height > 0:
    print(indic_s2.head(3))
```

### 2.3 Key EDA Findings (from existing analysis)

| Metric | Value | Implication |
|--------|-------|-------------|
| Train S1 entities | ~1.2M | Reference set |
| Train S2+S3 corpus | ~9.9M | Blocking target |
| Test S1 entities | ~1.73M | Prediction target |
| Test S2+S3 corpus | ~10.3M | Blocking target |
| Distinct tokens (train) | 2.06M | High cardinality → need DF pruning |
| Positive rate (blocked) | ~4.4% | Heavy class imbalance |
| Singleton rate | 5.6% | Must predict empty for F₀.₅ |
| Cross-script (Indic) | ~15% | Transliteration essential |
| Test countries | US, India, **France** | Open-set country handling |

---

## 3. PIPELINE ARCHITECTURE DEEP DIVE

### 3.1 Overall Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│  train_source1.tsv  train_source2.tsv  train_source3.tsv           │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│                  ┌──────────────────┐                               │
│                  │  normalize.py    │  ← Unicode, transliterate,    │
│                  │  norm_store.py   │     tokenize, state/code      │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Columnar Store  │  ← mmap'd .npy arrays         │
│                  │  (S2+S3 corpus)  │     ~3.2 GB train, ~3.5 GB test│
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Token DF Table  │  ← doc freq per token         │
│                  │  (build_token_df)│     df > 12K = drop           │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Inverted Index  │  ← part-sorted, packed        │
│                  │  (build_index)   │     uint64 (key40|doc24)      │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Blocking/Retrieval│ ← key_budget=14 rarest keys │
│                  │  (retrieve)      │     per_key_cap=80, top_k=100 │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Pair Features   │  ← 31 features (numba +       │
│                  │  (pair_features) │     rapidfuzz)                │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  LightGBM Train  │  ← ~7.2M pairs, 4.4% pos      │
│                  │  + Threshold     │     F₀.₅ macro opt τ=0.770    │
│                  └──────────────────┘                               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                        INFERENCE PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│  test_source1.tsv  test_source2.tsv  test_source3.tsv              │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│                  (Same normalization + store build)                │
│                             ▼                                       │
│                  ┌──────────────────┐                               │
│                  │  Reuse Token DF  │  ← rebuild or reuse           │
│                  │  + Inverted Index│                               │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│                  ┌──────────────────┐                               │
│                  │  Stream S1 in    │  ← batch=100K entities        │
│                  │  Batches         │     retrieve → features →     │
│                  │                  │     predict → write TSV       │
│                  └────────┬─────────┘                               │
│                           ▼                                         │
│              output/matching_results.tsv  (leaderboard)            │
│              output/candidate_pairs.tsv   (blocking audit)         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Core Modules Explained

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `normalize.py` | Text cleaning, Indic→Latin transliteration, hashing | `normalize_text()`, `transliterate()`, `tokens()`, `h64()` |
| `norm_store.py` | Streaming columnar store builder + mmap reader | `build_store()`, `Store` class |
| `blocking.py` | Token DF, key generation, inverted index, retrieval | `TokenDF`, `record_keys()`, `Index`, `retrieve()` |
| `pair_features.py` | 31 pair features (token overlap + string sim) | `FeatureBuilder.features()` |
| `pipeline.py` | Orchestration: corpus+index build, batch processing | `build_corpus_and_index()`, `process_batch()` |
| `collect_train.py` | Generate labeled training pairs with same blocking | Main entry for training data prep |
| `train_model.py` | LightGBM training + F₀.₅ threshold tuning | Main entry for model training |
| `predict_test.py` | **Main inference entry** → writes both TSVs | Main entry for test prediction |
| `run_outputs.py` | Streaming TSV writer | `write_submission()` |

### 3.3 Blocking Strategy Details

**Key Types Generated per Record (≤20 keys):**

| Key Type | Prefix | Description | DF Handling |
|----------|--------|-------------|-------------|
| Exact token | `t` | Core name/addr tokens | Drop if df > 12K; scope by state if df > 400 |
| 5-char prefix | `p` | First 5 chars of tokens ≥6 chars | State-scoped |
| Consonant skeleton | `k` | Vowel-dropped (typos/transliteration) | State-scoped |
| Name+Name composite | `nn` | Top 2 rare name tokens + state | Always generated |
| Name+Addr composite | `na` | Top rare name + addr token + state | Always generated |
| Addr+Addr composite | `aa` | Top 2 rare addr tokens + state | Always generated |
| First addr token | `a1` | Leading addr token + state | Always generated |
| Addr+Number | `ad` | Addr token + numeric code + state | Always generated |
| Compact name | `c8/c7/c5` | Space-free name prefix/suffix | No DF check |

**Retrieval Scoring:**
```
score(s1, candidate) = Σ (1/√df(key)) for shared keys
```
Top 100 candidates per S1 entity by this score.

---

## 4. RUNNING THE BASELINE END-TO-END

### 4.1 Full Training (If Retraining Needed)

```bash
# 1. Collect training pairs (blocking + features + labels)
cd student_resource
python code/business_entity_resolution/src/collect_train.py \
    --n-s1 200000 --topk 60 --batch 30000 --workers 8

# Output: work/cand/train_chunks/chunk_*.npz (~7.2M pairs)

# 2. Train LightGBM + tune threshold
python code/business_entity_resolution/src/train_model.py \
    --chunks work/cand/train_chunks \
    --holdout-frac 0.15 --max-neg-ratio 8.0 --rounds 800 \
    --model-out work/model.txt

# Output: work/model.txt, work/threshold.npy
# Expected: holdout macro F₀.₅ ≈ 0.772 at τ=0.770
```

### 4.2 Full Test Inference (Produces Submission Files)

```bash
# From student_resource/
python code/business_entity_resolution/src/predict_test.py \
    --model work/model.txt \
    --threshold work/threshold.npy \
    --batch 100000 \
    --topk 100 \
    --key-budget 14 \
    --per-key-cap 80 \
    --workers 8 \
    --tag _full \
    --out-dir output

# Expected runtime: 2-4 hours on r6g.4xlarge
# Output: output/matching_results.tsv, output/candidate_pairs.tsv
```

### 4.3 Fast Smoke Test (5 Minutes)

```bash
# Test on 1000 S1 entities with 100K corpus limit
python code/business_entity_resolution/src/predict_test.py \
    --model work/model.txt \
    --threshold work/threshold.npy \
    --batch 1000 \
    --topk 50 \
    --limit-corpus 100000 \
    --limit-s1 1000 \
    --tag _smoke \
    --out-dir output_smoke

# Validate
python utils/validate_submission.py \
    --matching output_smoke/matching_results.tsv \
    --candidate output_smoke/candidate_pairs.tsv \
    --test-dir dataset/test
```

### 4.4 Monitoring Progress

Open a second terminal and watch:
```bash
# Disk usage
watch -n 10 'du -sh work/store/ output/'

# Memory/CPU
htop

# Logs (in Jupyter, outputs print to cell)
tail -f nohup.out  # if running with nohup
```

---

## 5. VALIDATION & SUBMISSION

### 5.1 Local Validation (Mandatory Before Submit)

```bash
cd student_resource
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

**Expected Output:**
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732xxx
  valid S2/S3 match IDs: 103xxxxx
  matching_results.tsv: 1732xxx rows (xxxxx empty, xxxxx non-empty).
  candidate_pairs.tsv: 1732xxx rows (xxxxx empty, xxxxx non-empty).
WARNING: ID-existence check is OFF...
PASS — no blocking issues found. Safe to submit.
```

### 5.2 Common Validation Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| "header has no TAB but contains commas" | CSV not TSV | Use `sep='\t'` in to_csv |
| "duplicate source1_entity_id row" | Bug in writer | Check `run_outputs.py` groupby logic |
| "repeated ID inside a list" | Duplicate candidates | `dedupe_pairs()` in blocking.py |
| "matched IDs not in test set" | Wrong IDs written | Ensure corpus store only has test S2/S3 |
| "required S1 entity missing" | Incomplete S1 iteration | Check batch loop covers all `s1.n` |

### 5.3 Create Final Submission Zip

```bash
cd student_resource
# Fill Documentation_template.md with your approach
# Then zip:
zip -r <team_name>_submission.zip \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md

# Verify zip structure
unzip -l <team_name>_submission.zip
```

---

## 6. ADVANCED TUNING FOR 99TH PERCENTILE

### 6.1 Current Performance Ceiling Analysis

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Blocking Recall (matchable) | 99.5% | 99.9% | +0.4% |
| Blocking Recall (all) | 94% | 98% | +4% |
| Model F₀.₅ (holdout) | 0.772 | 0.85+ | +0.08 |
| Candidates/S1 entity | 100 | 50-80 | -20 to -50% |

### 6.2 High-Impact Tuning Knobs

#### A. Blocking Parameters (in `predict_test.py` / `blocking.py`)

```python
# Current defaults (in predict_test.py):
--topk 100              # Max candidates per S1 → try 80, 120
--key-budget 14         # Rarest keys to expand → try 16, 12
--per-key-cap 80        # Max postings per key → try 100, 60

# In blocking.py (rebuild index after changes):
MAX_KEYS = 20           # Keys per record → try 24
RARE_DF = 400           # State-scope threshold → try 300, 500
DF_THETA = 12_000       # Drop threshold → try 10_000, 15_000
```

**Experiment Design:**
```bash
# Grid search (run each as separate smoke test)
for topk in 80 100 120; do
  for key_budget in 12 14 16; do
    for per_key_cap in 60 80 100; do
      python predict_test.py --topk $topk --key-budget $key_budget \
          --per-key-cap $per_key_cap --limit-s1 5000 --tag "tune_${topk}_${key_budget}_${per_key_cap}" \
          --out-dir output_tune
      # Evaluate on holdout (need eval script)
    done
  done
done
```

#### B. Model Hyperparameters (in `train_model.py`)

```python
params = {
    "objective": "binary",
    "learning_rate": 0.06,        # Try 0.03, 0.05, 0.08
    "num_leaves": 96,             # Try 64, 128, 256
    "min_data_in_leaf": 60,       # Try 30, 100, 200
    "feature_fraction": 0.85,     # Try 0.7, 0.9, 1.0
    "bagging_fraction": 0.85,     # Try 0.7, 0.9
    "bagging_freq": 1,
    "lambda_l2": 1.0,             # Try 0.5, 2.0, 5.0
    "lambda_l1": 0.0,             # Add L1: try 0.1, 0.5
    "min_gain_to_split": 0.0,     # Try 0.01, 0.1
    "max_depth": -1,              # Try 8, 12, 16
    "verbose": -1,
    "num_threads": 16,            # Match vCPU count
    "metric": "average_precision",
}
```

#### C. Threshold Optimization

```python
# In train_model.py, expand sweep range:
for t in np.arange(0.01, 0.99, 0.01):  # Finer grid
    f = macro_f05(pv >= t, ...)
    
# Also try per-country thresholds (France may need different τ)
```

#### D. Feature Engineering Additions

Add to `pair_features.py` `_assemble()`:

```python
# 1. Phonetic similarity (Metaphone/Double Metaphone)
from rapidfuzz import fuzz
# metaphone_ratio = cd(la_n, lb_n, scorer=fuzz.METAPHONE, workers=w)

# 2. Character n-gram Jaccard (3-grams)
# 3. Embedding similarity (if using sentence-transformers, but >8B params!)
# 4. Numeric token overlap (PIN codes, phone numbers)
# 5. Acronym expansion matching
# 6. Address component-wise matching (street, city, state separate)
```

#### E. Ensemble / Multi-Model

```python
# Train 3 models with different seeds, average predictions
models = [lgb.Booster(model_file=f"work/model_seed_{s}.txt") for s in [0, 42, 123]]
preds = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
```

### 6.3 France-Specific Improvements (Test-Only Country)

```python
# In normalize.py - add French address patterns
FR_STOP_WORDS = {
    "rue", "avenue", "boulevard", "place", "chemin", "impasse",
    "allee", "voie", "quai", "route", "autoroute", "boulevard",
    "nord", "sud", "est", "ouest", "centre", "ville", "quartier"
}

# In blocking.py - ensure FR_REGIONS used for state scoping
# Already present: FR_REGIONS in normalize.py

# In pair_features.py - add country-specific feature weights
country_match_weight = {"US": 1.0, "India": 1.0, "France": 1.5}  # Boost France
```

### 6.4 Singleton Detection Enhancement

```python
# In predict_test.py - add singleton classifier
# Train a separate binary classifier: "has any match" vs "singleton"
# Use features: candidate count, max block score, name uniqueness, etc.
# If P(singleton) > 0.9, force empty prediction regardless of pair scores
```

---

## 7. TROUBLESHOOTING & MONITORING

### 7.1 Common Issues

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| OOM (Out of Memory) | Corpus store too large | Use `--limit-corpus`, increase instance RAM, or add swap |
| Slow blocking | Index not mmap'd | Ensure `mmap_mode="r"` in `Index.load()` |
| Low recall | DF_THETA too aggressive | Lower `DF_THETA` to 8000-10000 |
| Low precision | Threshold too low | Increase threshold, add negative mining |
| French entities all empty | No French training data | Verify country code 3 handling, add FR regions |
| Validation fails | TSV formatting | Check `sep='\t'`, no quoting, UTF-8 |

### 7.2 Memory Optimization

```bash
# Add swap if needed (emergency)
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Monitor
watch -n 5 'free -h && df -h /'
```

### 7.3 Profiling Commands

```python
# In notebook - profile blocking
import cProfile, pstats
cProfile.run('retrieve(index, tokdf, ...)', 'blocking.prof')
p = pstats.Stats('blocking.prof').sort_stats('cumulative')
p.print_stats(20)

# Profile feature computation
cProfile.run('fb.features(ps, pd, sc)', 'features.prof')
```

### 7.4 Key Logs to Watch

```
# During predict_test.py:
test: 1732xxx S1 entities, 103xxxxx corpus records (XXXs)
  [100000/1732xxx] pairs=XXXXX kept=XXXXX (XXXs)
  ...
DONE pairs=XXXXX kept=XXXXX (XXXXs)

# pairs = total candidates considered
# kept = pairs above threshold (final matches)
# kept/pairs ≈ 4-5% typical
```

---

## 🎯 QUICK REFERENCE: COMMAND CHEAT SHEET

```bash
# ===== FULL PIPELINE =====
# 1. Train (if needed)
python code/business_entity_resolution/src/collect_train.py --n-s1 200000 --topk 60
python code/business_entity_resolution/src/train_model.py --rounds 800

# 2. Predict (full test)
python code/business_entity_resolution/src/predict_test.py --batch 100000 --topk 100

# 3. Validate
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids

# ===== SMOKE TESTS =====
python predict_test.py --limit-corpus 100000 --limit-s1 1000 --tag _smoke --out-dir output_smoke

# ===== DEBUGGING =====
# Check corpus stats
python -c "
from norm_store import Store
st = Store('work/store/test_corpus_full')
print(f'Records: {st.n:,}')
print(f'Countries: {np.unique(st.country)}')
print(f'States: {len(np.unique(st.state))}')
"

# Check index stats
python -c "
from blocking import Index
idx = Index.load('work/store/test_full_index', n_parts=8)
print(f'Parts: {len(idx.parts)}, Postings: {idx.total_postings():,}')
"

# Check model
python -c "
import lightgbm as lgb
m = lgb.Booster(model_file='work/model.txt')
print(f'Trees: {m.num_trees()}, Best iter: {m.best_iteration}')
print(f'Features: {m.feature_name()}')
"
```

---

## 📈 PERFORMANCE TARGETS FOR 99TH PERCENTILE

| Stage | Current | Target | Action |
|-------|---------|--------|--------|
| Blocking recall (matchable) | 99.5% | 99.8%+ | Lower DF_THETA, add skeleton keys, more composites |
| Candidate reduction | 100/S1 | 50-70/S1 | Increase per_key_cap selectivity, better scoring |
| Model AUC | ~0.92 | 0.95+ | More features, ensemble, better neg sampling |
| Threshold F₀.₅ | 0.772 | 0.85+ | Per-country thresholds, singleton classifier |
| Inference time | 3 hrs | <2 hrs | Optimize numba, reduce top_k, parallelize batches |

---

## 📝 NEXT STEPS CHECKLIST

- [ ] Run smoke test to verify environment
- [ ] Run full validation on current outputs
- [ ] If PASS: submit current matching_results.tsv to leaderboard
- [ ] Analyze holdout errors (false pos/neg by country, name type)
- [ ] Implement top 3 tuning experiments from Section 6
- [ ] Re-run full inference with best config
- [ ] Validate & submit improved version
- [ ] Fill Documentation_template.md with final methodology
- [ ] Create final submission zip

---

**Remember:** The candidate_pairs.tsv size matters for final ranking! Smaller candidate sets with same/higher F₀.₅ rank higher. Optimize `top_k`, `key_budget`, `per_key_cap` for the best reduction ratio while maintaining recall ceiling > 99%.

Good luck! 🚀