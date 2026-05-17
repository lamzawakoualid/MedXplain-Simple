"""
MedXplain-Simple — Local JSON storage for patient reports and prior medical images.

Provides a lightweight, file-based storage layer with flexible schema
for storing analysis results, reports, and image references.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MedXplainDB:
    """Lightweight JSON-based local database for MedXplain-Simple.

    Stores analysis history, patient reports, and prior image references
    in structured JSON files.
    """

    def __init__(self, db_dir: str | Path = "medxplain_db/data"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.analyses_file = self.db_dir / "analyses.json"
        self.reports_file = self.db_dir / "reports.json"
        self.patients_file = self.db_dir / "patients.json"

        self._init_files()

    def _init_files(self) -> None:
        for f in (self.analyses_file, self.reports_file, self.patients_file):
            if not f.exists():
                f.write_text("[]")

    def _read(self, path: Path) -> list[dict]:
        with open(path) as f:
            return json.load(f)

    def _write(self, path: Path, data: list[dict]) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def save_analysis(self, analysis: dict[str, Any]) -> str:
        """Save an analysis result. Returns the generated record ID."""
        record_id = str(uuid.uuid4())[:8]
        record = {
            "id": record_id,
            "timestamp": time.time(),
            **analysis,
        }
        data = self._read(self.analyses_file)
        data.append(record)
        self._write(self.analyses_file, data)
        logger.info("Saved analysis %s", record_id)
        return record_id

    def get_analyses(
        self,
        patient_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """Retrieve analysis records, optionally filtered by patient ID."""
        data = self._read(self.analyses_file)
        if patient_id:
            data = [d for d in data if d.get("patient_id") == patient_id]
        return sorted(data, key=lambda x: x.get("timestamp", 0), reverse=True)[:limit]

    def save_report(self, report_meta: dict[str, Any]) -> str:
        """Save report metadata. Returns the record ID."""
        record_id = str(uuid.uuid4())[:8]
        record = {
            "id": record_id,
            "timestamp": time.time(),
            **report_meta,
        }
        data = self._read(self.reports_file)
        data.append(record)
        self._write(self.reports_file, data)
        return record_id

    def get_reports(self, patient_id: Optional[str] = None) -> list[dict]:
        data = self._read(self.reports_file)
        if patient_id:
            data = [d for d in data if d.get("patient_id") == patient_id]
        return sorted(data, key=lambda x: x.get("timestamp", 0), reverse=True)

    def save_patient(self, patient: dict[str, Any]) -> str:
        """Register or update patient metadata."""
        data = self._read(self.patients_file)
        existing = next((p for p in data if p.get("id") == patient.get("id")), None)
        if existing:
            existing.update(patient)
        else:
            data.append(patient)
        self._write(self.patients_file, data)
        return patient.get("id", "unknown")

    def get_patients(self) -> list[dict]:
        return self._read(self.patients_file)
