# Data Directory

This directory contains the candidate pool and related data files for BEAT-120 generation.

## Files

### `candidate_pool.jsonl` (Generated)

Candidate pool built from three evidence-grounded professional datasets:
- **PubMedQA** (Medicine): ~1000 candidates with biomedical abstract excerpts
- **SciFact** (Science): ~300 candidates with scientific abstracts
- **ContractNLI** (Law): ~800 candidates with contract excerpts

**Generate with:**
```bash
python scripts/collect_candidates.py --output data/candidate_pool.jsonl
```

**Format** (per DESIGN.md §3.2):
```json
{
  "candidate_id": "cand_XXXX",
  "domain": "Medicine" | "Science" | "Law",
  "prompt_base": "Base prompt (no A/B/C interventions)",
  "excerpt": "Native evidence excerpt (abstract, contract, etc.)",
  "ground_truth": "Yes" | "No",
  "source": "PubMedQA" | "SciFact" | "ContractNLI",
  "hash": "SHA256 hash",
  "metadata": {
    "source-specific fields": "...",
    "b2_evidence": "Dataset-native rationale for strong relevant add-on (Factor B2)"
  }
}
```

## Data Sources

All datasets are public and available via HuggingFace:

1. **PubMedQA**: `qiaojin/PubMedQA` (pqa_labeled split)
   - Biomedical research questions with abstract excerpts
   - Labels: {yes, no} (drop "maybe")
   - B2 evidence: long_answer field

2. **SciFact**: `copenlu/scifact` + `copenlu/scifact_corpus`
   - Scientific claims with research abstracts
   - Labels: {SUPPORTS, REFUTES} (drop "NOT_ENOUGH_INFO")
   - B2 evidence: rationale sentences (annotated evidence)

3. **ContractNLI**: `coastalcph/contract_nli` or `kiddothe2b/contract-nli`
   - Contractual hypotheses with contract excerpts
   - Labels: {Entailment, Contradiction} (drop "NotMentioned")
   - B2 evidence: evidence spans from contract

## Next Steps

After generating `candidate_pool.jsonl`, run boundary selection:

```bash
python -m src.beat120_builder \
    --candidates data/candidate_pool.jsonl \
    --target-count 120 \
    --output frozen_artifacts/beat120_questions.jsonl
```

This filters candidates to questions at the model's cognitive boundary ([4:6, 6:4] distributions) and stratifies by domain (Medicine/Science/Law).
