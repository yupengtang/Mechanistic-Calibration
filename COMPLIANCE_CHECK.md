# Code Compliance Check with DESIGN.md

**Date:** 2026-01-19  
**Status:** ✓ FULLY COMPLIANT (with 1 critical fix applied)

---

## Summary

The codebase has been systematically verified against all requirements in DESIGN.md. All core components are correctly implemented, with one critical issue identified and fixed.

---

## Critical Fix Applied

### 1. SSI 2.0 Drift Adjustment (analyze_results.py)

**Issue:** Behavioral SSI was missing drift adjustment term required by DESIGN.md §8.3.

**DESIGN.md requirement:**
```
SSI^(beh) = logit(P_rev | A3, B0) - logit(P_rev | A1, B0) 
            - [logit(P_rev | A0) - logit(P_stable | A0)]
```

**Previous implementation:**
```python
ssi_beh = logit(p_rev_a3) - logit(p_rev_a1)  # Missing drift adjustment!
```

**Fixed implementation:**
```python
ssi_beh_adjusted = ssi_beh_raw - drift_adjustment
where drift_adjustment = logit(P_rev_A0) - logit(P_stable_A0)
```

**Impact:** High - this ensures SSI captures social influence beyond natural instability, as theoretically motivated in §8.3.

---

## Verified Components (All Compliant)

### 1. Two-Pass Protocol (DESIGN.md §5.1)
✓ **Pass 1 baseline:** Run once per (question, model, temperature, replicate)  
✓ **Pass 2 interventions:** Reuse shared Pass 1 for all A/B/C conditions  
✓ **Implementation:** `run_experiment.py` lines 362-396

### 2. Factor Registry (DESIGN.md §5.2)
✓ **Reputation (A0-A3):** Implemented with evidence-free constraints  
✓ **Evidence (B0-B3):** Domain-specific templates for B2, placebo pools for B3  
✓ **Framing (C1-C2):** Rational vs deference framing  
✓ **Anti-confound policies:** Blacklist + forbidden_regex checks  
✓ **Implementation:** `prompts/registry.json`, `experiment_runner.py` lines 107-200

### 3. B2/B3 Evidence Add-ons (DESIGN.md §5.2)
✓ **B2 (relevant):** Uses dataset-native artifacts (long_answer, rationale_sentences, evidence_spans)  
✓ **B3 (placebo):** Samples from unrelated items within same domain  
✓ **Provenance logging:** addon_hash and addon_source_id tracked  
✓ **Length matching:** B2/B3 target same token length (±2 tokens)  
✓ **Implementation:** `experiment_runner.py` lines 144-181, `run_experiment.py` lines 331-337

### 4. Mechanistic Probing (DESIGN.md §8.1-8.2)
✓ **Tokenization robustness:** Aggregates variants via logsumexp  
✓ **Log-odds computation:** Separate for Yes/No, then diff  
✓ **Three-way taxonomy:** Deep / Superficial / Latent / Stable  
✓ **Drift-calibrated threshold:** 95th percentile of |ΔLogOdds| under A0  
✓ **Implementation:** `mechanistic_probing.py` lines 64-122, 187-221

### 5. Logging Schema (DESIGN.md §11.1)
✓ **Identifiers:** trial_id, question_id, domain, question_hash, excerpt_hash  
✓ **Metadata:** model_id, provider, timestamp_utc, commit_hash  
✓ **Harness:** temperature, top_p, max_tokens, stop_sequences, replicate_id  
✓ **Conditions:** reputation_factor, evidence_factor, framing_factor  
✓ **Pass results:** raw_output, parsed_label, truncation flags, token usage, latency  
✓ **Mechanistic:** lp_yes, lp_no, logodds, delta_logodds, variants_aggregated  
✓ **Padding:** target_token_len, actual_token_len, padding_id  
✓ **Evidence:** excerpt_hash, addon_hash, addon_source_id  
✓ **Taxonomy:** revision_category (computed in analysis)  
✓ **Implementation:** `data_structures.py` lines 61-109

### 6. AEP Transformation (DESIGN.md §4.4)
✓ **Domain-aware:** Medicine/Science/Law specific entity mappings  
✓ **50/50 split:** SHA256-based deterministic selection  
✓ **Both fields:** Applied to prompt_base AND excerpt  
✓ **Metadata tracking:** is_aep flag, aep_entities_replaced  
✓ **Implementation:** `run_experiment.py` lines 276-295, `aep_transformation.py`

### 7. SSI 2.0 (DESIGN.md §8.3) [FIXED]
✓ **Behavioral SSI:** Now includes drift adjustment  
✓ **Mechanistic SSI:** E[ΔLogOdds | A3, B0] - E[ΔLogOdds | A1, B0]  
✓ **Alignment diagnostic:** Correlation ρ(SSI^beh, SSI^mech) computation  
✓ **Scaling hypothesis:** Testable via regression on model size  
✓ **Implementation:** `analyze_results.py` lines 165-232

### 8. Model Set (DESIGN.md §6)
✓ **Mechanistic core:** Llama 8B/70B, Qwen 7B/32B  
✓ **Extended:** Llama 405B, Qwen 72B  
✓ **Behavioral anchors:** Claude 3.5 Sonnet, GPT-4o, Gemini Pro 1.5, Mistral Large  
✓ **Reasoning case study:** o1-preview, o3-mini, gpt-4o-mini (minimal protocol)  
✓ **Implementation:** `frozen_artifacts/models.json`

### 9. Decoding Harness (DESIGN.md §7)
✓ **Core:** T ∈ {0.0, 0.7}, top_p=1.0, R=3  
✓ **Extended:** T ∈ {0.0, 0.3, 0.7, 1.0}, top_p ∈ {0.9, 1.0}, R=5  
✓ **Stop sequences:** ["</s>", "###", "User:", "Assistant:", "Q:", "A:", "B:"]  
✓ **Output contract:** First line Yes/No, then 4-6 sentences  
✓ **Implementation:** `experiment_runner.py`, `config.json`

### 10. Length Matching (DESIGN.md §5.5)
✓ **Tolerance:** ±2 tokens  
✓ **Reference tokenizer:** API models use config.api_reference_tokenizer_model_id  
✓ **Per-condition padding:** Neutral filler to match longest condition  
✓ **Logging:** target_token_len, actual_token_len, padding_id  
✓ **Implementation:** `prompt_utils.py` lines 15-78

---

## Minor Documentation Fixes

### DESIGN.md Section References
Fixed incorrect cross-references:
- §4.2 → §5.2 (Factor Registry)
- §4.5 → §5.5 (Length Matching)
- §4.4 → §5.4 (Opponent Construction)
- §7.2 → §8.2 (Taxonomy)

These were off by 1-2 due to section restructuring during theoretical enhancement.

---

## Code Quality Checks

✓ **Syntax validation:** All Python files pass `py_compile`  
✓ **Import structure:** Correct relative imports throughout `src/`  
✓ **Optional dependencies:** torch/transformers gracefully handled  
✓ **Thread safety:** Output lock for concurrent writes  
✓ **Error handling:** Try-except blocks for API calls and probing

---

## Remaining Notes

1. **Environment setup:** Users need to install dependencies via `requirements.txt` and set up `.env` with `OPENROUTER_API_KEY`
2. **GPU availability:** Local mechanistic models gracefully skipped if torch not available
3. **Data pipeline:** `scripts/collect_candidates.py` correctly sources from PubMedQA, SciFact, ContractNLI
4. **Frozen artifacts:** All pre-registered files in `frozen_artifacts/` follow DESIGN.md specs

---

## Conclusion

**The codebase is now fully compliant with DESIGN.md after applying the SSI drift adjustment fix.**

All experimental components (two-pass protocol, factorial design, mechanistic probing, logging, analysis) correctly implement the theoretical framework and pre-registered procedures specified in the research proposal.
