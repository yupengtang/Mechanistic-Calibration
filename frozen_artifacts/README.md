# Frozen Artifacts

This directory contains **pre-registered artifacts** that are frozen before data collection to ensure reproducibility and minimize researcher degrees of freedom.

## Files

### 1. `beat120_questions.jsonl`

**Status**: To be generated and frozen before main experiment

The BEAT-120 benchmark: 120 evidence-grounded questions at models' cognitive boundary ([4:6, 6:4] answer distribution) across Medicine/Science/Law.

**Format**:
```json
{
  "question_id": "q001",
  "domain": "Medicine",
  "prompt_base": "Base prompt (no A/B/C interventions)",
  "excerpt": "Native evidence excerpt (abstract/contract snippet)",
  "hash": "SHA256 hash (first 16 chars)"
}
```

**Generation procedure**: See `src/beat120_builder.py`

### 2. `models.json`

**Status**: Frozen

Model registry defining:
- Mechanistic probing set (local models with logits)
- Behavioral anchor set (API models)
- Reasoning case study models (minimal protocol)
- Screening models (for question selection)

**Versioning**: If models become unavailable, document substitutions in `models_substitutions.log`

### 3. `analysis_plan.md`

**Status**: Frozen

Pre-registered statistical analysis plan:
- Primary outcome (reversal)
- GLMM specification
- Secondary outcomes (taxonomy, SSI)
- Exclusion criteria
- Multiple comparison corrections
- Effect size reporting

**Deviations**: Any deviations from this plan must be documented in the final paper with justification.

### 4. Sample Files

- `beat120_questions_SAMPLE.jsonl`: Example questions for testing (5 samples)

**Note**: Actual BEAT-120 should have 120 questions. Use sample for development/testing only.

## Workflow

### Before Data Collection

1. **Generate BEAT-120**:
   ```bash
   python -m src.beat120_builder \
       --candidates data/candidate_pool.jsonl \
       --output frozen_artifacts/beat120_questions.jsonl
   ```

2. **Verify frozen artifacts**:
   ```bash
   python -c "
   import json
   from pathlib import Path
   
   # Check all frozen artifacts
   assert Path('frozen_artifacts/beat120_questions.jsonl').exists()
   assert Path('frozen_artifacts/models.json').exists()
   assert Path('frozen_artifacts/analysis_plan.md').exists()
   assert Path('prompts/registry.json').exists()
   
   # Verify BEAT-120 has 120 questions
   with open('frozen_artifacts/beat120_questions.jsonl') as f:
       questions = [json.loads(line) for line in f]
   assert len(questions) == 120, f'Expected 120 questions, got {len(questions)}'
   
   print('All frozen artifacts verified')
   "
   ```

3. **Commit and tag**:
   ```bash
   git add frozen_artifacts/ prompts/
   git commit -m "Freeze artifacts for BEAT-120 v1.0"
   git tag -a v1.0-frozen -m "Frozen artifacts before data collection"
   git push origin v1.0-frozen
   ```

### During Data Collection

**DO NOT MODIFY** any frozen artifacts. If issues are discovered:

1. Document in `frozen_artifacts/issues.log`
2. Continue with current artifacts (if feasible)
3. Plan for post-hoc robustness checks if needed

### After Data Collection

Frozen artifacts remain unchanged. Extensions or modifications should:

1. Create new versions (`beat120_v2_questions.jsonl`)
2. Clearly mark as post-hoc in publications
3. Compare with original frozen version

## Integrity Checks

Generate checksums for verification:

```bash
# Generate SHA256 checksums
sha256sum frozen_artifacts/*.jsonl frozen_artifacts/*.json > frozen_artifacts/checksums.txt

# Verify (run before analysis)
sha256sum -c frozen_artifacts/checksums.txt
```

## Contact

Questions about frozen artifacts? Open an issue with tag `[frozen-artifacts]`.

