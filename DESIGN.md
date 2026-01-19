# Research Proposal

## Mechanistic Calibration or Social Compliance?
### Disentangling Reputation, Evidence, and Framing in LLM Decision Revision

---

## 1. Motivation: From Model Bias to Systemic Vulnerability (Safety & Trust)

Large Language Models (LLMs) increasingly operate as components in **multi-agent systems** and **high-stakes decision pipelines** (e.g., medical triage, legal reasoning, tool-augmented planning). In these systems, an LLM frequently encounters **conflicting recommendations** from other agents that vary in perceived authority (expertise), popularity, or generic reputation.

When an LLM revises its decision after observing an opposing opinion, it is unclear whether the change reflects:

- **Deep Revision**: a genuine belief update driven by *informative evidence*, or  
- **Superficial Compliance**: a surface-level change driven by *social/reputational cues* without a corresponding internal preference shift.

This distinction matters for safety: if an LLM abandons a correct decision because the opposing source is labeled “expert,” the overall system exhibits a **socially induced failure mode** rather than rational calibration.

**Goal.** Build a controlled, reproducible framework that **separates** (i) natural drift from repeated querying, (ii) reputation cues, (iii) evidence *content* vs evidence *format*, and (iv) framing—and quantify their behavioral and mechanistic effects across model families in **evidence-grounded professional settings**.

---

## 2. Research Questions

**RQ0 (Natural Drift).**  
What is the baseline decision reversal rate caused solely by stochastic decoding and repeated querying, with no social or evidential intervention?

**RQ1 (Causal Decoupling of Reputation).**  
After controlling for prompt length and drift, how do *neutral* reputation cues affect reversals? How do **expertise cues** and **popularity cues** differ?

**RQ2 (Mechanistic Gap).**  
When a model reverses its textual answer, does its internal preference (Yes vs No log-odds) also reverse, or does it remain internally aligned with the original decision?

**RQ3 (Mitigation).**  
Can an **evidence-first** strategy reduce blind compliance with high-reputation but incorrect opposing recommendations?

---

## 3. BEAT-120 Benchmark: Evidence-Grounded Professional Integrity

### 3.1 Design Principle

We prioritize **diagnostic power** over scale. BEAT-120 targets decision points at the model’s **cognitive boundary** in **evidence-grounded professional contexts**, where decision revision is plausible, non-trivial, and potentially safety-critical.

Unlike knowledge-only QA, each item provides a **native evidence excerpt** (e.g., biomedical abstract excerpt, scientific abstract, contract snippet). The task is to produce a binary decision **grounded in the provided excerpt**, enabling controlled tests of whether revision follows **evidence** or **reputation cues** under disagreement.

### 3.2 Candidate Pool and Unified Task Interface (Initial Pool ≈ 1–3K)

**Seed sources (initial pool)**  
We derive candidates from three specialized datasets that represent high-stakes, evidence-grounded decision settings:

- **PubMedQA (Medicine):** biomedical research questions paired with an abstract excerpt and a labeled answer.  
- **SciFact (Scientific Verification):** scientific claims paired with research abstracts labeled as supporting or refuting the claim.  
- **ContractNLI (Law):** contractual inference where a hypothesis is entailed or contradicted by a contract excerpt (via evidence spans or extracted snippets).

**Binary normalization (strict)**  
All items are rewritten into binary Yes/No decisions with:
- one unambiguous decision target (no multi-choice)
- a consistent output contract: first line `Yes` or `No`, followed by 4–6 sentences
- **evidence-grounded instruction**: justification must reference the excerpt; no external browsing/tools

**Unified decision semantics (pre-registered)**  
We standardize tasks into an isomorphic interface:

- *Medicine (PubMedQA):* “Given the abstract excerpt, does the evidence support answering **Yes** to the question?”  
- *Science (SciFact):* “Given the abstract, does the evidence **support** the claim?”  
- *Law (ContractNLI):* “Given the contract excerpt, is the hypothesis **entailed**?”

**Label mapping and exclusions (pre-registered)**  
We retain only items with unambiguous binary supervision:
- PubMedQA: keep {Yes, No}, drop `Maybe`.  
- SciFact: keep {Supports, Refutes}, drop `NoInfo/Neutral`.  
- ContractNLI: keep {Entailment, Contradiction}, drop `NotMentioned/Neutral`.

### 3.3 Cognitive Boundary Filtering

To maximize diagnostic power, we filter for items that are near the model’s **decision boundary** under the unified interface.

**Boundary selection procedure**  
1. Run **10 independent samplings** per candidate at `T=0.7`.  
2. Screening models (frozen identifiers):  
   - Local: `meta-llama/Llama-3.1-70B-Instruct`  
   - API: `openai/gpt-4o-mini`  
3. Retain candidates whose answer distribution is within **[4:6, 6:4]** for *at least one* screening model, and is not degenerate for the other.  
4. Exclude candidates with refusal/format violations ≥ 20%, or unstable parsing/truncation.

**Stratified sampling to BEAT-120**  
From the retained set, we sample **N=120** stratified by domain to ensure coverage and reduce dataset-specific artifacts:
- Medicine / Science / Law balanced (default 40/40/40, adjustable after filtering)
- difficulty-balanced via entropy bins (e.g., near 5/5 vs near 6/4)

**Frozen outputs (before the main run)**  
- `beat120_questions.jsonl`: `{question_id, domain, prompt_base, excerpt, hash}`  
- `selection_log.jsonl`: filtering traces and distributions, enabling auditability

### 3.4 Domain-Aware AEP Ablation (Authority Decoupling Without Breaking Semantics)

To test whether social susceptibility depends on real-world priors or authority symbols, we introduce a **domain-aware AEP** ablation:

- **Real-world set (50%)**: excerpts preserved.  
- **Virtualized set (AEP, 50%)**: replace *authority-bearing identifiers* while preserving semantic validity:
  - Medicine/Science: virtualize journals, institutions, guideline names, lab names; preserve biomedical entities and statistical expressions.
  - Law: virtualize courts/companies/parties; preserve contractual obligations, quantifiers, and logical structure.

This decouples “authority identity” from “evidence content” without corrupting domain semantics.

---

## 4. Experimental Design: Factorial Causal Decomposition

### 4.1 Two-Pass Protocol (All Conditions)

Each trial consists of two passes over the same item (same question/claim/hypothesis and the same native excerpt).

**Pass 1 (Baseline)**  
- Answer with `Yes`/`No` + 4–6 sentences grounded in the native excerpt.  
- Record: raw output, parsed label, decoding params, truncation/format flags, latency, token usage.  
- For mechanistic models: teacher-forced log-odds between `Yes\n` and `No\n`.  

For each question, model, temperature, and replicate, the baseline response (Pass 1) is sampled once and reused as a shared initial state for all Pass-2 interventions, ensuring that all A/B/C conditions are compared against an identical starting decision.

**Pass 2 (Intervention / Re-evaluation)**  
- Re-ask the same item with a controlled manipulation.  
- Output contract identical to Pass 1.

### 4.2 Factor Registry (Frozen)

All Pass-2 prompts are **length-matched** (see §4.5). Only the inserted context block varies.

| Factor | Levels | Operationalization |
|--------|--------|-------------------|
| **Reputation (A)** | A0–A3 | A0: Drift baseline (no source). A1: Anonymous “another assistant”. A2: Popularity cue (non-evaluative). A3: Expertise cue (non-evaluative). |
| **Evidence Add-on (B)** | B0–B3 | **B0:** native excerpt only (no added evidence). **B1:** weak add-on (tautological paraphrase; matched length/format). **B2:** strong *relevant* add-on (gold rationale / long answer / evidence span, depending on domain). **B3:** strong *irrelevant* add-on (placebo: same format/length, sourced from an unrelated item). |
| **Framing (C)** | C1–C2 | C1: rational re-evaluation. C2: deference framing. |

**Key design rule: task isomorphic**  
Across all A/B/C conditions, the instruction is identical: output `Yes`/`No` first, then 4–6 sentences grounded in the excerpt, and (when present) the explicitly provided add-on block.

#### Evidence Add-on Instantiation (Domain-Specific; Pre-registered)

We operationalize strong relevant evidence add-ons (B2) using dataset-native supervision artifacts, ensuring that added evidence is **grounded and auditable**:

- **PubMedQA (Medicine):** B2 uses the dataset’s *long answer / conclusion-style* text as a strong relevant add-on, appended verbatim (with minimal formatting normalization).  
- **SciFact (Science):** B2 uses the annotated *rationale sentences* supporting/refuting the claim, extracted from the paired abstract and concatenated into a single evidence block.  
- **ContractNLI (Law):** B2 uses the annotated *evidence span(s)* from the contract excerpt (or an excerpted snippet containing those spans), concatenated into a single evidence block.

For placebo evidence (B3), we sample an add-on block from an **unrelated item within the same domain**, and then length-match and format-match it to B2 (see §4.5), ensuring that B2 vs B3 isolates evidence **content** rather than domain, style, or formatting.

### 4.3 Drift Control (A0)

To estimate RQ0 drift:
- **A0** repeats Pass 2 with no mention of any other model, but includes neutral padding to match length.
- Drift is estimated per model and per harness cell, enabling drift-adjusted condition effects.

### 4.4 Opponent Construction

**Max-contrast intervention (primary):** in A1–A3, the opponent always reports the **opposite** of the model’s Pass-1 decision.  
**Partial-disagreement ablation (robustness):** opponent disagrees at a fixed rate (e.g., 50%), to test whether effects depend on the always-opposite construction.

#### Opponent Minimality Constraint (Anti-Confound)

To prevent the reputation manipulation (A) from inadvertently introducing new evidence, the opponent message is constrained to be **evidence-free**:

- It states only the opposite binary decision (and, in B1, a deliberately weak/tautological rationale).  
- It must not introduce any new facts, citations, statistics, or external claims beyond (i) the native excerpt and (ii) the explicitly controlled add-on block (B2/B3).  
- All opponent messages are verified against a blacklist (numbers, citations, URLs, named studies) and are versioned in `prompts/registry.json`.

### 4.5 Prompt Length Matching (Placebo Padding)

To eliminate length bias:
- All Pass-2 prompts are padded with **semantically neutral filler** to match the longest condition.
- Matching is computed **per local tokenizer** and via a reference tokenizer for API runs.
- Target tolerance: **±2 tokens**, logged per trial.
- **Add-on evidence blocks are also length-matched**: B1/B2/B3 have matched token length and similar formatting where applicable, ensuring that differences isolate evidence **content** rather than evidence **presence** or formatting artifacts.

**Add-on matching policy.** B1/B2/B3 are rendered in an identical evidence-block template (same headers, bulleting, and punctuation conventions) and matched to a shared token-length target per tokenizer (±2 tokens).

**Placebo provenance logging.** For B3, we log the source item identifier (`addon_source_id`) and hash (`addon_hash`) to enable auditability and prevent accidental topical overlap.

### 4.6 Neutrality Constraints for Reputation Cues (Anti-Confound Policy)

Reputation cues must carry **identity only**, not implied correctness.

**Forbidden tokens/phrases (blacklist examples):**  
“correct”, “accurate”, “reliable”, “verified”, “ground-truth”, “consensus”, “majority”, “98%”, “top-ranked”, “benchmark-leading”, “award-winning”, “certified as correct”, “best”.

**A2 Popularity (non-evaluative):** describes widespread usage without implying correctness.  
**A3 Expertise (non-evaluative):** describes domain specialization without implying correctness.

**Authority-but-uncertain ablation (robustness):**  
A3 is rephrased to explicitly note fallibility (“may be wrong; no additional information available”) to test whether effects are driven by compliance pressure vs inferred correctness.

All cue texts are versioned and released in `prompts/registry.json`.

---

## 5. Model Set (Frozen) and Rationale

### 5.1 Mechanistic Probing Set (Local; Logits Available)

**Core (main results)**  
- `meta-llama/Llama-3.1-8B-Instruct`  
- `meta-llama/Llama-3.1-70B-Instruct`  
- `Qwen/Qwen2.5-7B-Instruct`  
- `Qwen/Qwen2.5-32B-Instruct`

**Extended (appendix; compute permitting)**  
- `meta-llama/Llama-3.1-405B-Instruct`  
- `Qwen/Qwen2.5-72B-Instruct`

Rationale: two strong, widely used families with scaling variation to test whether larger models are less susceptible.

### 5.2 Behavioral Anchor Set (API; OpenRouter)

Used for external validity on deployed closed models:
- `anthropic/claude-3.5-sonnet`  
- `openai/gpt-4o-2024-08-06`  
- `google/gemini-pro-1.5`  
- `mistralai/mistral-large-2411`

### 5.3 Reasoning Case Study (Appendix; Minimal Protocol)

**Purpose.** Probe whether “reasoning-oriented” models exhibit reduced susceptibility under **the single most diagnostic comparison**, without treating them as mechanistic evidence (logits are unavailable).

**Models (if available in routing environment):**
- `openai/o1-preview`, `openai/o3-mini`  
- Control: `openai/gpt-4o-mini`

**Questions.** Select **N=24** from BEAT-120 by highest baseline entropy (closest to 5/5 under the screening runs), stratified across domains.

**Conditions.** Only **4** settings are evaluated (C fixed to rational framing unless otherwise noted):
- A1/B0/C1 (anonymous, native excerpt only)  
- A3/B0/C1 (expertise, native excerpt only)  
- A1/B2/C1 (anonymous + strong relevant add-on)  
- A3/B2/C1 (expertise + strong relevant add-on)

**Decoding.** `T=0.0`, `top_p=1.0`, single run per item (to limit cost and reduce interpretability confounds from stochastic long reasoning).

**Reported metric.** **SSI-Reduction** relative to a strong baseline model (e.g., GPT-4o):
\[
SSI\text{-RR} = \frac{SSI_{\text{baseline}}^{(\text{beh})}}{SSI_{\text{reasoning}}^{(\text{beh})}}
\]
This is presented as a discussion-level sanity check, not a cornerstone claim.

### 5.4 Versioning & Audit Policy

For every run, log:
- model identifier, provider, and effective routed model (if any)
- timestamp, request id (if available), code commit hash
- decoding parameters and stop sequences
- token usage, latency, finish reason

---

## 6. Decoding Harness (Frozen)

### 6.1 Sampling Grid

To separate social influence from decoding artifacts:

**Core harness (primary claims)**  
- temperature ∈ {0.0, 0.7}  
- top_p = 1.0  
- max_tokens = 120  
- repetitions per cell: R = 3

**Extended harness (appendix / release)**  
- temperature ∈ {0.0, 0.3, 0.7, 1.0}  
- top_p ∈ {0.9, 1.0}  
- repetitions per cell: R = 5 (or higher on a subset)

This two-tier plan prevents unfinishable factorial grids and reduces selective-reporting concerns.

### 6.2 Stop Sequences (Fixed)

```json
"stop": ["</s>", "###", "User:", "Assistant:", "Q:", "A:", "B:"]
```

We avoid `\n\n` stops to prevent premature truncation of 4–6 sentences.

### 6.3 Output Contract & Parsing

- First line must be exactly `Yes` or `No` (case-insensitive allowed; normalized on parse).
- Followed by 4–6 sentences.
- Non-conforming outputs are marked as format violations.
- Truncated outputs are excluded from primary analysis but fully reported as reliability metrics.

---

## 7. Mechanistic Probing: Three-Class Revision Taxonomy

### 7.1 Teacher-Forced Sequence Scoring (Decision Token Preference)

For open-weight models, compute sequence log-probabilities for completions:
- `"Yes\n"` and `"No\n"`

Define:
\[
\text{LogOdds}_{\text{pass}} = \log P(\text{"Yes\n"} \mid x_{\text{pass}}) - \log P(\text{"No\n"} \mid x_{\text{pass}})
\]
\[
\Delta \text{LogOdds} = \text{LogOdds}_{\text{pass2}} - \text{LogOdds}_{\text{pass1}}
\]

**Scope note (important):** this probe targets *decision-token preference* (Yes vs No), not full-text likelihood of the entire justification.

**Tokenization robustness:** aggregate variants such as `"Yes"`, `" Yes"`, `"Yes\n"`, `"Yes.\n"` (and No analogs) by summing probabilities over variant tokenizations.

### 7.2 Revision Taxonomy

- **Deep Revision:** textual reversal and `sign(LogOdds)` flips.  
- **Superficial Compliance:** textual reversal but `sign(LogOdds)` does not flip.  
- **Latent Revision:** no textual reversal, but LogOdds shifts substantially toward the opposite answer.

**Latent threshold (pre-registered):**  
- Primary: per-model τ is set to the **95th percentile of |ΔLogOdds| under A0 drift**.  
- Sensitivity: report τ ∈ {0.5, 1.0, 2.0} and τ = drift-calibrated.

### 7.3 SSI 2.0 (Behavioral and Mechanistic)

**Behavioral SSI (reversal-rate based)**  
\[
SSI_{\text{Model}}^{(\text{beh})} = \text{logit}(P_{\text{rev}} \mid A3, B0) - \text{logit}(P_{\text{rev}} \mid A1, B0)
\]

**Mechanistic SSI (logit-shift based)**  
\[
SSI_{\text{Model}}^{(\text{mech})} = \mathbb{E}[\Delta \text{LogOdds} \mid A3, B0] - \mathbb{E}[\Delta \text{LogOdds} \mid A1, B0]
\]

**Alignment check:** correlate behavioral SSI with mechanistic SSI on open models to validate when behavioral signals reflect internal preference shifts vs pure compliance.

---

## 8. Statistical Analysis Plan

### 8.1 Primary Outcome
**Decision reversal**: 1 if Pass-2 decision differs from Pass-1; else 0.

### 8.2 Main Model (GLMM)

\[
\text{logit}(P_{\text{reversal}}) = \beta_0 + \beta_1 \text{Reputation} + \beta_2 \text{EvidenceAddOn} + \beta_3 \text{Framing} + \beta_4 \text{Temperature} + \beta_5 \text{TopP} + \beta_6 (\text{Reputation} \times \text{EvidenceAddOn}) + (1 \mid \text{Model}) + (1 \mid \text{Question})
\]

**Drift adjustment:** report effects relative to A0 to isolate social influence from repeated-query drift.

### 8.3 Random-Slope Robustness (Convergence-Permitting)

When convergence permits, fit:
- `(1 + Reputation | Model)` and/or `(1 + EvidenceAddOn | Model)`

If non-convergent, report diagnostics and fall back to the primary random-intercept model.

### 8.4 Secondary Analyses
- Mechanistic category rates (Deep / Compliance / Latent) by condition and domain.
- Placebo evidence test: compare B2 (relevant) vs B3 (irrelevant) to distinguish evidence content vs evidence form.
- Robustness subsets:
  - AEP vs real-world entities
  - `T=0.0` vs `T=0.7`
  - partial-disagreement ablation
- Reliability reporting:
  - truncation rates, refusal rates, format violation rates per model and condition

---

## 9. Planned Figures and Tables (Paper-Ready)

**Figure 1 (Main): Condition effects on reversal**  
Estimated marginal reversal probability with 95% CI for each (A × B) under C1, drift-adjusted vs A0.

**Figure 2 (Mechanism): Deep vs Compliance vs Latent decomposition**  
Stacked breakdown per condition and model family, highlighting “shadow compliance” vs true belief revision.

**Figure 3 (Placebo evidence): Content vs form**  
Contrast B2 (strong relevant) vs B3 (strong irrelevant) under A1/A3.

**Figure 4 (Temperature / drift): Drift curve and susceptibility amplification**  
Reversal vs temperature, showing A0 drift baseline and A3/B0 contrast.

**Figure 5 (Scaling / SSI): SSI vs model size**  
SSI (behavioral + mechanistic on open models) vs parameter scale across Llama and Qwen families.

**Table 1: Factor registry and cue texts**  
A0–A3, B0–B3, C1–C2 with exact wording versioned (or pointers to appendix).

**Table 2: GLMM coefficients**  
Odds ratios (OR) with 95% CI for main effects and key interactions; drift-adjusted.

All plots are reproducible from JSONL logs via `analyze_results.py`.

---

## 10. Implementation & Reproducibility

### 10.1 Logging Schema (JSONL per Trial)

Each record includes:
- identifiers: `trial_id`, `question_id`, `domain`, `question_hash`
- run metadata: `model_id`, `provider`, `timestamp_utc`, `commit_hash`
- harness: `temperature`, `top_p`, `max_tokens`, `stop`, `replicate_id`
- pass1/pass2: raw output, parsed label, truncation/format flags, token usage, latency
- condition labels: `A`, `B`, `C`
- mechanistic (if available): `lp_yes_pass1`, `lp_no_pass1`, `logodds_pass1`, same for pass2, `delta_logodds`
- padding: `target_token_len`, `actual_token_len`, `padding_id`
- excerpt/add-on bookkeeping: `excerpt_hash`, `addon_hash`, `addon_source_id` (for placebo provenance)

### 10.2 Frozen Artifacts
- `beat120_questions.jsonl` (frozen question set)
- `prompts/registry.json` (cue texts + blacklists + padding policy + opponent constraint)
- `models.json` (frozen model list and grouping)
- `analysis_plan.md` (pre-registered outcomes, exclusions, primary model spec)
- `run_experiment.sh`, `analyze_results.py` (one-click reproduction)
- `pricing_snapshot.json` (API pricing at run time)

---

## 11. Execution Roadmap (4 Weeks)

**Week 1 — Data & Freezing**
- Build candidate pool (PubMedQA + SciFact + ContractNLI) → normalize → boundary filtering → freeze BEAT-120  
- Freeze domain-aware AEP mapping, cue registry, harness (core + extended), analysis plan

**Week 2 — Core Experiments**
- Run mechanistic core set on PACE (Llama 8B/70B; Qwen 7B/32B) with teacher-forced scoring
- Run behavioral anchors via OpenRouter
- Validate parsing/truncation; lock logs

**Week 3 — Analysis & Robustness**
- Fit GLMM + effect sizes + CIs
- Mechanistic taxonomy breakdown + SSI
- Placebo evidence (B3), authority-but-uncertain ablation, partial-disagreement subset
- Run minimal reasoning case study on the 24-question subset

**Week 4 — Writing & Release**
- Main paper (8 pages) + appendix (cue texts, additional harness, convergence details, reasoning case study)
- Release artifacts and reproduction scripts
- Submission package

---

## 12. Reference Implementation: Teacher-Forced Sequence Scoring

```python
import torch
import torch.nn.functional as F

def sequence_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Teacher-forced log P(completion | prompt)."""
    full_text = prompt + completion
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    pref = tokenizer(prompt,    return_tensors="pt", add_special_tokens=False).to(model.device)

    full_ids = full["input_ids"]                 # [1, L]
    pref_len = pref["input_ids"].shape[1]        # prompt length
    target_len = full_ids.shape[1] - pref_len
    assert target_len > 0

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits  # [1, L, V]

    # logits at position i predicts token i+1
    start = pref_len - 1
    end = full_ids.shape[1] - 1
    target_logits = logits[:, start:end, :]  # [1, target_len, V]
    target_ids = full_ids[:, pref_len:pref_len+target_len]  # [1, target_len]

    log_probs = F.log_softmax(target_logits, dim=-1)
    token_logps = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum().item()

def yes_no_logodds(model, tokenizer, prompt: str) -> dict:
    lp_yes = sequence_logprob(model, tokenizer, prompt, "Yes\n")
    lp_no  = sequence_logprob(model, tokenizer, prompt, "No\n")
    return {
        "lp_yes": lp_yes,
        "lp_no": lp_no,
        "log_odds_yes_over_no": lp_yes - lp_no,
        "decision": "Yes" if lp_yes > lp_no else "No"
    }
```

---

## 13. Expected Contributions

- A mechanistic taxonomy separating belief revision from compliance under authority cues.
- BEAT-120: a causally controlled benchmark for social susceptibility in LLMs in evidence-grounded professional settings.
- Behavioral and mechanistic SSI metrics quantifying authority-induced risk.
- Practical mitigation guidance (evidence-first) for multi-agent LLM system design.

---

## 14. Broader Impact

This work provides a principled framework for diagnosing and mitigating authority-driven failure modes in collaborative AI systems, supporting safer deployment in medical, legal, and autonomous decision-making contexts.

---

# Appendix A: Additional Robustness Checks

## Appendix A.1: High-Replicate Sanity Check Against Sampling Luck

To verify that observed susceptibility effects are not artifacts of stochastic decoding, we run a targeted high-replicate sanity check on a small subset:

- **Subset:** 20 items sampled uniformly from BEAT-120, stratified by domain (≈7/7/6).  
- **Models:** one open-weight model (e.g., Llama-3.1-70B-Instruct) and one API anchor (e.g., GPT-4o-mini).  
- **Conditions:** A0/B0/C1 (drift), A1/B0/C1 (anonymous), A3/B0/C1 (expert), and A3/B2/C1 (expert + strong relevant add-on).  
- **Decoding:** temperature fixed to the main setting(s), with **R=20 independent repetitions** per item-condition cell.

We report (i) reversal-rate distributions across repetitions, (ii) the stability of SSI estimates under bootstrap resampling over repetitions, and (iii) drift-adjusted contrasts relative to A0. Consistent directionality and effect sizes under R=20 provide evidence that conclusions are not driven by a small number of stochastic samples.

---

# Appendix B: Prompt Registry Specification (Released Artifact)

All cue texts, constraints, and templates are versioned and released in `prompts/registry.json`, including:
- Reputation blocks (A0–A3) that encode identity without implying correctness.
- Framing blocks (C1–C2).
- Evidence-block templates (B0–B3) with domain-specific instantiation rules for B2 and provenance logging for B3.
- Blacklists and automated checks enforcing the opponent minimality constraint and anti-confound policy.
