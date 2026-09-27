# Data & Model Artifacts

**This repository contains only code.** Large data files and model artifacts are stored externally due to GitHub size limits.

## 📥 Download Instructions

### 1. Training Data (~2.5 GB)
Place in `student_resource/dataset/train/`:

| File | Size | Download |
|------|------|----------|
| `train_source1.tsv` | 210 MB | [Google Drive / S3 / HF Link] |
| `train_source2.tsv` | 489 MB | [Google Drive / S3 / HF Link] |
| `train_source3.tsv` | 504 MB | [Google Drive / S3 / HF Link] |
| `train_ground_truth.tsv` | 127 MB | [Google Drive / S3 / HF Link] |

### 2. Test Data (~1.2 GB)
Place in `student_resource/dataset/test/`:

| File | Size | Download |
|------|------|----------|
| `test_source1.tsv` | 175 MB | [Google Drive / S3 / HF Link] |
| `test_source2.tsv` | 510 MB | [Google Drive / S3 / HF Link] |
| `test_source3.tsv` | 506 MB | [Google Drive / S3 / HF Link] |

### 3. Model Artifacts (Pre-trained)
Place in `student_resource/work/`:

| File | Size | Download |
|------|------|----------|
| `model.txt` | ~7 MB | [Google Drive / S3 / HF Link] |
| `threshold.npy` | <1 MB | [Google Drive / S3 / HF Link] |
| `store/train_full_index.part*.npy` | ~1.4 GB | [Google Drive / S3 / HF Link] |
| `store/train_full_tokdf.*.npy` | ~25 MB | [Google Drive / S3 / HF Link] |
| `store/test_corpus_full/` | ~1.5 GB | [Google Drive / S3 / HF Link] |
| `store/test_full_index.part*.npy` | ~1.4 GB | [Google Drive / S3 / HF Link] |

## 🚀 Quick Start After Download

```bash
# 1. Clone repo
git clone https://github.com/Bindu020705/amazon.git
cd amazon

# 2. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 3. Download data (see links above) and place in:
#    student_resource/dataset/train/
#    student_resource/dataset/test/
#    student_resource/work/

# 4. Run inference
cd student_resource
python code/business_entity_resolution/src/predict_test.py --batch 100000 --topk 100

# 5. Validate
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

## 🔄 Regenerating Artifacts (If Needed)

If you don't download pre-built artifacts, the pipeline rebuilds them:

```bash
# Build train corpus + index (first run only, ~30 min)
python code/business_entity_resolution/src/collect_train.py --n-s1 200000 --topk 60

# Train model (~20 min)
python code/business_entity_resolution/src/train_model.py --rounds 800

# Build test corpus + index (during inference, auto)
# First predict_test.py run will build test index automatically
```

## 📊 Expected Outputs

After running `predict_test.py`:
- `output/matching_results.tsv` (~2 MB) — **Leaderboard submission**
- `output/candidate_pairs.tsv` (~52 MB) — **Blocking audit**

## 📦 Final Submission Package

```bash
# Fill Documentation_template.md first, then:
cd student_resource
zip -r <team_name>_submission.zip \
    output/matching_results.tsv \
    output/candidate_pairs.tsv \
    code/business_entity_resolution/ \
    Documentation_template.md
```

---

## 🔗 Suggested Hosting

| Platform | Free Tier | Best For |
|----------|-----------|----------|
| **Google Drive** | 15 GB | Simple sharing |
| **Hugging Face Datasets** | Unlimited | ML datasets, versioned |
| **AWS S3** | 5 GB (12 mo) | Production pipelines |
| **GitHub Releases** | 2 GB/file | Model weights only |
| **Kaggle Datasets** | Unlimited | Competition data |

**Recommendation:** Upload to **Hugging Face Datasets** (free, versioned, citable) and update links above.

---

*Replace `[Google Drive / S3 / HF Link]` with actual URLs after uploading.*