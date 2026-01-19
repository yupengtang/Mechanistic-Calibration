"""
BEAT-120 Experimental Framework
Mechanistic Calibration or Social Compliance?
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .data_structures import (
    Question,
    PassResult,
    TrialRecord,
    ExperimentConfig,
    load_questions,
    save_questions,
    read_jsonl_records,
    write_jsonl_record
)

_MECH_AVAILABLE = False
try:
    # Optional dependency: torch
    from .mechanistic_probing import (
        yes_no_logodds,
        classify_revision,
        compute_drift_calibrated_threshold,
        LogOddsResult,
        RevisionTaxonomy
    )
    _MECH_AVAILABLE = True
except Exception:
    # Allow importing src without torch installed (e.g., for docs, data prep, or API-only runs).
    pass

from .experiment_runner import TwoPassExperiment

from .prompt_utils import (
    compute_length_matched_prompts,
    verify_length_matching
)

from .aep_transformation import (
    apply_aep_transformation,
    should_apply_aep,
    transform_question_to_aep,
    AEPMetadata
)

__all__ = [
    # Data structures
    "Question",
    "PassResult",
    "TrialRecord",
    "ExperimentConfig",
    "load_questions",
    "save_questions",
    "read_jsonl_records",
    "write_jsonl_record",
    
    # Mechanistic probing (optional; available only if torch is installed)
    *(
        [
            "yes_no_logodds",
            "classify_revision",
            "compute_drift_calibrated_threshold",
            "LogOddsResult",
            "RevisionTaxonomy",
        ]
        if _MECH_AVAILABLE
        else []
    ),
    
    # Experiment runner
    "TwoPassExperiment",
    
    # Prompt utilities
    "compute_length_matched_prompts",
    "verify_length_matching",
    
    # AEP transformation
    "apply_aep_transformation",
    "should_apply_aep",
    "transform_question_to_aep",
    "AEPMetadata",
]

