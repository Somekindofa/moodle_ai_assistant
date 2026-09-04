"""Tests that annotation_to_documents emits correct cohort metadata.

Two layers of access control are covered here:

1. The project-level control — an annotation whose project sets
   ``allowed_cohort_id`` is restricted to that cohort; ``None`` means open.
2. The craft-level safety net (``CRAFT_COHORT_MAP``) — content tagged with a
   craft belonging to a partner organisation stays restricted even when nobody
   set a project cohort. This exists because legacy/bulk-imported annotations
   carry no project at all, and because a forgotten project setting must not
   silently publish confidential material.
"""

from unittest.mock import patch

from config.settings import ConfigurationManager
from services.annotation_service import AnnotationService


def _make_service():
    return AnnotationService(ConfigurationManager())


def _make_annotation(cohort_id=None, craft=""):
    return {
        "annotation_id": 1,
        "video_id": 10,
        "video_filename": "demo.mp4",
        "video_filepath": "/videos/demo.mp4",
        "start_time": 0.0,
        "end_time": 5.0,
        "audio_filepath": "",
        "source_type": "local",
        "project_name": "Test",
        "craft": craft,
        "annotation_created_at": "2026-01-01",
        "annotation_updated_at": "2026-01-01",
        "transcription": "Hello world",
        "extended_transcript": None,
        "allowed_cohort_id": cohort_id,
    }


# ── The query must actually select the column, or everything below is moot ──

def test_base_query_selects_allowed_cohort_id():
    assert "allowed_cohort_id" in AnnotationService._BASE_QUERY


# ── Project-level control ───────────────────────────────────────────────

def test_open_annotation_has_open_access_true():
    svc = _make_service()
    docs = svc.annotation_to_documents(_make_annotation(cohort_id=None), use_extended=False)
    assert docs
    assert docs[0].metadata["open_access"] is True
    assert docs[0].metadata["cohort_id"] == -1


def test_restricted_annotation_has_cohort_id():
    svc = _make_service()
    docs = svc.annotation_to_documents(_make_annotation(cohort_id=7), use_extended=False)
    assert docs
    assert docs[0].metadata["open_access"] is False
    assert docs[0].metadata["cohort_id"] == 7


# ── Craft-level safety net ──────────────────────────────────────────────

def test_mapped_craft_is_restricted_when_project_sets_no_cohort():
    svc = _make_service()
    with patch.dict("services.annotation_service.CRAFT_COHORT_MAP",
                    {"lv_rivetage_maletterie": 30}, clear=True):
        docs = svc.annotation_to_documents(
            _make_annotation(cohort_id=None, craft="lv_rivetage_maletterie"),
            use_extended=False,
        )
    assert docs
    assert docs[0].metadata["open_access"] is False
    assert docs[0].metadata["cohort_id"] == 30


def test_unmapped_craft_stays_open():
    svc = _make_service()
    with patch.dict("services.annotation_service.CRAFT_COHORT_MAP",
                    {"lv_rivetage_maletterie": 30}, clear=True):
        docs = svc.annotation_to_documents(
            _make_annotation(cohort_id=None, craft="glassblowing"),
            use_extended=False,
        )
    assert docs
    assert docs[0].metadata["open_access"] is True
    assert docs[0].metadata["cohort_id"] == -1


def test_explicit_project_cohort_wins_over_craft_map():
    svc = _make_service()
    with patch.dict("services.annotation_service.CRAFT_COHORT_MAP",
                    {"lv_rivetage_maletterie": 30}, clear=True):
        docs = svc.annotation_to_documents(
            _make_annotation(cohort_id=7, craft="lv_rivetage_maletterie"),
            use_extended=False,
        )
    assert docs
    assert docs[0].metadata["open_access"] is False
    assert docs[0].metadata["cohort_id"] == 7


# ── The map must survive being read at import time ──────────────────────

def test_craft_cohort_map_falls_back_to_dotenv_file(monkeypatch):
    """The module-level map is built at import, before any ConfigurationManager
    exists to call load_dotenv() — so when the variable is absent from the
    process environment it must consult the .env file itself.

    Regression guard: a loader that only reads os.environ silently yields {} in
    production, which would publish partner content as open-access.

    (conftest.py deliberately stubs dotenv and blocks .env reads, because the
    test user is ACL-denied from the real secrets file — so the fallback is
    asserted through a patched dotenv_values rather than the real file.)
    """
    from services import annotation_service

    monkeypatch.delenv("CRAFT_COHORT_MAP", raising=False)
    monkeypatch.setattr(
        annotation_service, "dotenv_values",
        lambda path: {"CRAFT_COHORT_MAP": '{"lv_rivetage_maletterie": 30}'},
    )

    loaded = annotation_service._load_craft_cohort_map()

    assert loaded == {"lv_rivetage_maletterie": 30}


def test_craft_cohort_map_survives_malformed_json(monkeypatch):
    """A corrupt map must not crash ingestion — but it must be loud, because
    the failure mode is silently open-access content."""
    from services import annotation_service

    monkeypatch.setenv("CRAFT_COHORT_MAP", "{not valid json")

    assert annotation_service._load_craft_cohort_map() == {}
