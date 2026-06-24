"""Tests that annotation_to_documents emits correct cohort metadata."""

from services.annotation_service import AnnotationService
from unittest.mock import patch, MagicMock


def _make_annotation(cohort_id=None):
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
        "annotation_created_at": "2026-01-01",
        "annotation_updated_at": "2026-01-01",
        "transcription": "Hello world",
        "extended_transcript": None,
        "allowed_cohort_id": cohort_id,
    }


def _make_service():
    from config.settings import ConfigurationManager
    with patch("config.settings.load_dotenv"), \
         patch("config.settings.dotenv_values", return_value={}), \
         patch("services.annotation_service.Path.exists", return_value=True):
        return AnnotationService(ConfigurationManager())


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
