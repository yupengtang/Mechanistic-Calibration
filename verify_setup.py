#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup Verification Script for BEAT-120 Project

This script performs a lightweight verification of:
- required files
- required Python dependencies
- API key presence (optional, only needed for OpenRouter models)

It intentionally avoids embedding any current date/time metadata.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def check_file(path: Path, description: str) -> bool:
    if path.exists():
        print(f"OK: {description}: {path}")
        return True
    print(f"MISSING: {description}: {path}")
    return False


def check_python_dependencies() -> bool:
    print("\nChecking Python dependencies")
    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "HuggingFace Datasets (PubMedQA/SciFact/ContractNLI)"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("statsmodels", "statsmodels"),
        ("dotenv", "python-dotenv"),
        ("requests", "requests"),
    ]

    ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"OK: {name}")
        except Exception:
            ok = False
            if module == "dotenv":
                print("MISSING: python-dotenv (pip install python-dotenv)")
            else:
                print(f"MISSING: {name} (pip install {module})")
    return ok


def check_api_key() -> bool:
    print("\nChecking API key (OpenRouter)")
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if key:
        print("OK: OPENROUTER_API_KEY is set in environment")
        return True

    env_path = Path(".env")
    if not env_path.exists():
        print("NOTE: .env not found. This is fine if you only run local models.")
        print("If you run OpenRouter models, create .env with OPENROUTER_API_KEY=...")
        return False

    try:
        # Minimal .env parsing without requiring python-dotenv
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                val = line.split("=", 1)[1].strip()
                if val:
                    print("OK: OPENROUTER_API_KEY found in .env")
                    return True
        print("NOTE: OPENROUTER_API_KEY not found in .env")
        return False
    except Exception as e:
        print(f"NOTE: Could not read .env: {e}")
        return False


def main() -> int:
    repo_root = Path(".")
    ok = True

    print("Checking required files")
    ok &= check_file(repo_root / "config.json", "Experiment configuration")
    ok &= check_file(repo_root / "prompts" / "registry.json", "Prompt registry")
    ok &= check_file(repo_root / "frozen_artifacts" / "models.json", "Frozen model set")
    ok &= check_file(repo_root / "frozen_artifacts" / "analysis_plan.md", "Analysis plan")
    ok &= check_file(repo_root / "requirements.txt", "Python requirements")

    ok &= check_python_dependencies()
    _ = check_api_key()

    if ok:
        print("\nRESULT: PASS")
        return 0
    print("\nRESULT: FAIL (see messages above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

