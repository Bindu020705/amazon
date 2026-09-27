# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** —  
**Team Members:** —  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We solve entity resolution as a two-stage **blocking + gradient-boosted classifier** pipeline built for scale: all 9.97M Source-2/Source-3 test records are normalized into a memory-mapped columnar store, a multi-key inverted index generates ≤100 scored candidates per Source-1 entity (a ~0.001% slice of the naive search space), and a LightGBM model over 31 string/structure features decides the final match set per entity with a precision-oriented threshold. The whole pipeline streams in batches, so peak RAM stays around one batch regardless of dataset size, and it runs end-to-end on CPU in a few hours.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training data established the facts that shaped the design:

* **Scale rules out pairwise comparison**: 1.73M test entities × 9.97M corpus records ≈ 1.7×10¹³ naive pairs; blocking must cut this by ~7 orders of magnitude.
* **Names are extremely high-cardinality**: 2,058,510 distinct normalized name/addr tokens across the train corpus. Exact-token keys alone have very low pair coverage.
* **True pairs share *rare* keys**: the median true pair shares a key with document frequency ≈ 2; ~90% of true pairs share a key with sample-df ≤ 100. This motivated weighting candidates by inverse document frequency instead of raw key hits.
* **Cross-script data**: many S2/S3 business names are written in Indic scripts (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi, Gujarati, Odia). We built a Unicode-block-driven transliterator so names can match their Latin-script variants.
* **Country shift**: the test set adds France (259k entities) with no French training data — the pipeline treats country as an open string label, normalizes French regions, and uses script/regex normalization rather than hard-coded country logic.
* **Class balance**: ~4.4% of blocked pairs are positives; 5.6% of S1 entities are singletons, so correctly outputting an empty match list is worth real metric mass (a singleton scores 1.0 iff predicted empty).

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage streaming pipeline)  
**Core Innovation:** A multi-key blocking scheme whose key inventory is *pruned by document frequency* (keys on tokens with df > 12,000 are discarded, state-scoped fallbacks keep locality), with candidate scoring by summed inverse-√df of shared keys, and a fully streaming disk-backed execution model (mmap'd columnar store, part-sorted inverted index, per-batch feature computation) that keeps the 8-core CPU-only run feasible.

---

## 3. Candidate Generation (Blocking)

* **Blocking keys used** (`record_keys`): exact normalized name tokens, 5-character name/addr prefixes, 6-character consonant-skeleton keys (name & address), composite keys (name+state, name+country, addr-token+name-token, first-name-token+addr-token), compact forms (name/addr strings truncated to 8/7/5 chars), and state-scoped variants of high-df keys. Each record emits ≤20 keys; any key whose corpus document frequency exceeds 12,000 is dropped, and keys with df > 400 are state-scoped.
* **Retrieval**: per entity, the `key_budget=14` rarest keys are selected; each key expands up to `per_key_cap=80` postings (rarest first); candidates are scored by the summed inverse-√df of the keys they share with the query and the **top-100 per entity** survive. Scoring candidates across multiple index parts is deduplicated by packing (s1, doc) into a single int64 and aggregating weights.
* **Candidate pairs generated**: ≤100 per Source-1 entity → at most ~173M candidates total for 1.73M entities (~0.001% of the naive space). The reduction ratio is ≥ 10⁵ relative to all-pairs.
* **How we ensured true matches were not lost:** probe experiments on training data measured a **blocking recall ceiling ≈ 0.94 including singletons (≈0.995 among entities with any true match)** — i.e., of entities that truly have matches, 99.5% have at least one true match inside the retrieved candidate set. The rarest-key-first expansion protects entities whose only shared signal is a rare token; skeleton keys catch typos and transliteration variants; composite/state-scoped keys recover high-frequency names. ~0.005% of true pairs share *no* blocking key — accepted residual loss, traded for a compact candidate file (smaller candidate sets rank higher in final evaluation).

---

## 4. Matching Model

**Features used (31 total, `pair_features.py`):**
- Name features: rare-token Jaccard and overlap statistics over *atomized* name tokens with corpus-derived rarity ranks (computed by a numba-parallel intersection kernel), rapidfuzz `normalized_levenshtein` / `token_set`-style similarities on the normalized name strings.
- Address features: the same rare-atom Jaccard/overlap family on normalized address tokens, address edit similarities, state agreement, address-token count ratios.
- Other: country agreement, legal-suffix match flags, token-count ratios, the blocking score itself, per-store token-rank statistics. All features are symmetric-compatible, float32, and computed in O(|atoms|) via the numba kernel.

**Model type:** LightGBM binary classifier (leaf-wise GBDT), trained on ~7.2M candidate pairs collected from 120k stratified training entities (~4.4% positives), using the identical streaming retrieval path as inference.  
**Threshold selection method:** decision threshold swept on a 15% held-out entity split, maximizing **macro F₀.₅ per entity** (precision 2× weighted, singletons included) → **τ = 0.770**. Pairs with p ≥ τ are written to `matching_results.tsv`; every retrieved candidate (post-top-k) is written to `candidate_pairs.tsv`, guaranteeing matches ⊆ candidates.

---

## 5. Results & Error Analysis

- **F₀.₅ Score (macro): 0.772** (holdout, threshold 0.770; singletons included in the macro average).
- **Common false positives (wrong merges):** same-franchise / generic-name businesses in the same city ("pizza hut" + address overlap), name matches with completely different street numbers, and Indic-script transliteration variants that normalize to the same skeleton but are genuinely different branches.
- **Common false negatives (missed matches):** entities whose true matches share only a very common key (all keys dropped by the df-pruning) — e.g., extremely generic names with short/generic addresses; and records where the S2/S3 entry lacks the state component so state-scoped keys never fire. The 0.5% of true pairs that share no blocking key are structurally unreachable.

---

## 6. Conclusion

A df-pruned multi-key inverted index plus an inverse-frequency-scored top-k retrieval brings a 1.7×10¹³ search space down to ≤100 candidates per entity with a ~0.995 recall ceiling on matchable entities, and a LightGBM model over 31 string/structure features with an F₀.₅-tuned threshold converts them into the final match lists at macro F₀.₅ = 0.772 on holdout. The biggest lessons: (1) token document-frequency pruning is the single highest-leverage blocking decision; (2) transliteration and skeleton keys are essential for cross-script, typo-laden data; (3) a fully streaming, mmap-backed design keeps the whole problem tractable on an 8-core CPU-only machine.

---

## Appendix

### A. Code Artefacts

Complete runnable code ships in `code/business_entity_resolution/`:

```
src/
  normalize.py      — text normalization, Indic→Latin transliteration, h64 hashing
  norm_store.py     — streaming columnar store builder (build_store) + mmap reader (Store)
  blocking.py       — TokenDF, record_keys, part-sorted inverted index, top-k retrieve
  pair_features.py  — 31-feature pair builder (numba kernel + rapidfuzz)
  pipeline.py       — corpus/index build orchestration + per-batch processing
  collect_train.py  — collects labeled training pairs with the same blocking path
  train_model.py    — trains LightGBM and tunes the F₀.₅ threshold
  run_outputs.py    — streaming writer for both submission TSVs
  predict_test.py   — ENTRY POINT: end-to-end test inference → output/*.tsv
README.md, requirements.txt
```

Reproduce the deliverables from `student_resource/`:

```bash
python code/business_entity_resolution/src/predict_test.py --batch 20000 --topk 100
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The pipeline rebuilds all derived artifacts (corpus store, token DFs, inverted index) deterministically from `dataset/test/*.tsv`; the trained model artifact (`work/model.txt`) is included in the package.

### B. Additional Results

* Threshold sweep (holdout macro F₀.₅): peak 0.7722 at τ=0.770; precision-heavy drop-off confirmed below τ≈0.65.
* Feature importance: rare-token name Jaccard ranks first, followed by name edit similarity and blocking score; country agreement and legal-suffix flags contribute marginally.
* Blocking ablations: removing skeleton keys costs ~1.5pt ceiling recall; removing state-scoped keys costs ~2pt; dropping the df-pruning explodes candidate counts ~40× for +0.3pt ceiling.
