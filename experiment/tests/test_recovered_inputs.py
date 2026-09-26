"""Check that released audited inputs can be read without model/API access."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment'))
from src.data_structures import load_questions

DATA = ROOT / 'reproducibility/data-repair/outputs'


class RecoveredInputLoading(unittest.TestCase):
    def test_all_released_datasets_load(self):
        for name, expected in [('original_benchmark_recovered.jsonl', 1989),
                               ('benchmark_evidence_complete_v2.jsonl', 1989),
                               ('controls_evidence_complete_v2.jsonl', 500)]:
            with self.subTest(name=name):
                questions = load_questions(DATA / name)
                self.assertEqual(len(questions), expected)
                self.assertTrue(all(q.ground_truth in ('Yes', 'No') for q in questions))
                self.assertTrue(all(q.provenance and q.input_sha256 for q in questions))

    def test_modified_input_is_rejected(self):
        with (DATA / 'benchmark_evidence_complete_v2.jsonl').open() as handle:
            row = json.loads(next(handle))
        row['excerpt'] += ' changed'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.jsonl'
            path.write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_questions(path)


if __name__ == '__main__':
    unittest.main()
