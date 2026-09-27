# Business Entity Resolution — Amazon ML Challenge 2026

Entity resolution across 3 noisy business-data sources (US / India / France): given a
Source-1 entity, decide which Source-2 / Source-3 records refer to the same real-world
business.

## Deliverables

* `output/matching_results.tsv` — one row per test Source-1 entity; tab-separated
  `source1_entity_id \t comma,separated,matched,ids` (empty match column = singleton).
* `output/candidate_pairs.tsv` — same layout with the full candidate list fed to the model.

## Pipeline

1. **Normalization** (`normalize.py`) — lowercasing, unicode cleanup, address-token
   extraction, state/region canonicalization, and a Unicode-name-derived Indic→Latin
   transliteration (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gurmukhi,
   Gujarati, Odia) so cross-script names can match.
2. **Corpus store** (`norm_store.py`) — S2+S3 records streamed into a memory-mappable
   columnar store (`build_store`); `Store` mmaps all arrays; IDs in row order in
   `ids.txt`. Peak RAM stays flat regardless of dataset size.
3. **Blocking** (`blocking.py`) — per-record keys (exact tokens, 5-char prefixes,
   6-char consonant skeletons, composites, state-scoped variants) packed into a sorted
   inverted index. Retrieval expands the `key_budget` rarest keys per entity
   (`per_key_cap` postings each) and scores candidates by summed inverse-sqrt-df of
   shared keys, keeping the top-k. Token DFs with a sample-based cutoff (`DF_THETA`)
   keep the vocabulary tractable (2M+ distinct tokens).
4. **Pair features** (`pair_features.py`) — 31 features per candidate pair: rare-token
   Jaccard/overlap statistics on names and addresses (computed with a numba parallel
   kernel over atom arrays), rapidfuzz similarity ratios, state/country agreement,
   token-count ratios, and the blocking score itself.
5. **Model** (`train_model.py`, LightGBM) — binary classifier trained on ~7.2M pairs
   from 120k train entities (~4.4% positive). Decision threshold tuned for the metric
   (macro F0.5 per entity, precision-weighted) on a 15% entity holdout.
6. **Inference** (`predict_test.py`) — builds the test corpus + index once, then
   streams Source-1 in batches of 100k: retrieve → features → predict → write both
   TSVs incrementally. Singletons (no candidate above threshold) write an empty match
   column.

## Reproducing

```bash
# full test inference (writes both output TSVs)
python code/business_entity_resolution/src/predict_test.py --batch 100000 --topk 100

# validate
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Useful flags: `--limit-corpus N --limit-s1 N --tag _slice --out-dir ...` for a fast
end-to-end smoke run on a slice.

## Results

Holdout macro **F0.5 = 0.772** at threshold 0.770 (singletons included in the metric).
