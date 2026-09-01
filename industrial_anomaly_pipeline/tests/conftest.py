"""
Ensures `industrial_anomaly_pipeline/` (not the repo root) is on sys.path,
so test modules can use the same absolute imports (`from ingestion... import
...`, `from analytics... import ...`) as `main.py`, regardless of the
directory pytest is invoked from.
"""
import sys
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
