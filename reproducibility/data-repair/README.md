# Verified input recovery and evidence-complete datasets

This package recovers all **1,989 historical question–excerpt pairs** and
provides separate evidence-complete inputs for new experiments. It does not
replace or relabel any historical model output.

## Choose the right file

| File in `outputs/` | Purpose |
|---|---|
| `original_benchmark_recovered.jsonl` | Historical inputs: 890 Medicine, 299 Science, 800 Law. Preserve the original excerpt construction, including its limitations. |
| `benchmark_evidence_complete_v2.jsonl` | Same questions and labels, with 48 Law excerpts expanded to retain all nonempty annotated evidence spans. For fresh experiments only. |
| `controls_evidence_complete_v2.jsonl` | Same 500 control questions and labels, with 16 Law excerpts repaired. The other 484 question–excerpt pairs are unchanged. For fresh experiments only. |
| `audit_report.json` | Coverage, verification levels, output checksums and correction counts. |
| `benchmark_evidence_changes.json`, `control_evidence_changes.json` | Per-item changes, old/new hashes, missing evidence spans and lengths. |
| `previous_recovery_prompt_corrections.json` | The 96 Science questions incorrectly associated with an abstract by the earlier recovery script, with their verified replacements. |

V2 item identifiers have a separate namespace. A changed excerpt invalidates its
previous model responses and generated opponent arguments. Run fresh Pass 1 and
all revision branches; regenerate the rationale stimuli for changed inputs.
Historical confidence tiers are cleared for changed excerpts. No model has yet
been evaluated on these repaired inputs by this package.

## Offline verification and regeneration

Python 3.9+ suffices; no third-party dependency, account, network connection,
GPU or API credit is required once `sources/` is present.

```bash
python repair_inputs.py
python -m unittest discover -s . -p 'test_*.py' -v
```

The script verifies all frozen source checksums before running. Existing output
files must be byte-identical; it refuses to overwrite differing artifacts.
To regenerate independently into a fresh directory:

```bash
python repair_inputs.py --out /path/to/new-output-directory
```

## Recovery evidence

- **1,987 items:** both the excerpt hash and the combined question–excerpt hash
  match the original Claude run. Question hashes are real hashes, contrary to
  the earlier recovery notes; an excerpt-only join is not sufficient.
- **q0405:** directly verified against its surviving frozen input. The original
  de-duplicated trial table supplies the original label and confidence tier.
- **q0828:** matched to the full question hash and excerpt hash in the selection
  log and the original trial label. Its question ID follows the selection order,
  independently cross-checked against every available logged item in the
  1,371-record surviving selection prefix. This is not a claimed match to a
  missing generation log.
- All **552** complete surviving frozen records match the recovered question,
  excerpt and label. All 29,778 retained trial rows agree on their recovered
  question ID, domain, ground-truth label and historical confidence tier.

The selection log contains full screening questions but only excerpt prefixes;
it is not itself an intact input backup. Public source text supplies the full
excerpts, which are validated using the saved hashes.

The historical Law loader formed a 200-character context window around each
annotated span, joined windows with `\n\n...\n\n`, and truncated to 2,000
characters. It also stripped the document before using its original offsets.
Reconstructing this behavior matches all 800 original Law inputs on both
hashes. Recovery deliberately preserves those bytes; evidence repair is a
different operation.

The old excerpt-only recovery file assigned the first matching Science claim
to every question sharing an abstract. The 96 corrected questions were in the
reconstructed input file, **not in the original saved model outputs**. Saved
output analyses are therefore unchanged; never use the superseded reconstructed
file to run Science experiments.

## Evidence repair policy

For every Law item, use the original source document and hypothesis identifier,
validate its label, and inspect each annotated evidence span independently.
Joining all spans into one string is not a valid containment test when the
excerpt retains intervening text. Whitespace-only annotations have no semantic
content and are explicitly counted in metadata.

Keep an existing excerpt if it contains every nonempty annotated span. Otherwise
take the contiguous source-document interval covering all annotated spans, with
200 characters of context on each side where available. Do not truncate that
interval to satisfy an arbitrary length cap. Preserve offsets, document hashes,
and source identifiers. This establishes evidence-span coverage, not independent
expert validation of every dataset label or argument.

The audit found 48/800 affected original Law items and 16/201 affected control
Law items. This does not by itself prove that their answers were wrong. Any
updated empirical claim about the repaired inputs requires new model calls.

## Source provenance and terms

`sources/manifest.json` records source and stored SHA-256 hashes. Compressed
sources retain the historical trial table, raw run, partial frozen input and
selection log, old reconstructed inputs and control inputs. These are audit
evidence, not alternative datasets to use accidentally. The captured normalizer
is used only for its pure conversion functions; its download functions are not
called. The source bundle includes the official ContractNLI license and terms.

- PubMedQA: `qiaojin/PubMedQA`, `pqa_labeled/train` (1,000 source records), from
  the cached Arrow dataset; source repository: https://github.com/pubmedqa/pubmedqa.
- SciFact: official corpus and train/dev claims from
  https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz;
  source repository: https://github.com/allenai/scifact.
- ContractNLI: official train/dev/test release, 607 source documents;
  source repository: https://github.com/stanfordnlp/contract-nli.

Third-party datasets retain their original terms; the code license does not
relicense them. No model weights or credentials are included. Initial capture
from the development workspace requires `pyarrow` to export the cached PubMedQA
records; ordinary offline regeneration does not.
