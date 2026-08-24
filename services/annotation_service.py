"""Service for managing video annotation database operations.

Video annotation data was migrated out of the standalone SQLite database into
Moodle's own MariaDB (local_videoelicit plugin tables) on 2026-02-19 — see
/opt/video_elicitation_annotation_tool/START_HERE_DATABASE_MIGRATION.md. This
service reads from the live mdl_local_videoelicit_* tables directly; the old
SQLite file is stale and no longer written to.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

import pymysql
import pymysql.cursors

from langchain_core.documents.base import Document

from config.settings import ConfigurationManager
from services import translation_service


logger = logging.getLogger(__name__)


class AnnotationService:
    """Service for reading video annotations from Moodle's MariaDB."""

    _BASE_QUERY = """
        SELECT
            a.id AS annotation_id,
            a.videoid AS video_id,
            a.starttime AS start_time,
            a.endtime AS end_time,
            a.audiofilepath AS audio_filepath,
            a.transcription,
            a.transcriptionstatus AS transcription_status,
            a.language AS language,
            a.craft,
            a.task,
            a.timecreated AS annotation_created_at,
            a.timemodified AS annotation_updated_at,
            v.filename AS video_filename,
            v.filepath AS video_filepath,
            v.duration AS video_duration,
            v.source_type,
            p.name AS project_name,
            p.description AS project_description
        FROM mdl_local_videoelicit_annotations a
        JOIN mdl_local_videoelicit_videos v ON a.videoid = v.id
        LEFT JOIN mdl_local_videoelicit_projects p ON v.projectid = p.id
        WHERE a.transcriptionstatus = 'completed'
    """

    def __init__(
        self,
        config_manager: ConfigurationManager,
        db_host: str = "localhost",
        db_user: str = "moodleuser",
        db_name: str = "moodle",
    ):
        self.config_manager = config_manager
        self.db_host = db_host
        self.db_user = db_user
        self.db_name = db_name
        self.db_password = os.getenv("MOODLE_DB_PASSWORD", "")
        if not self.db_password:
            logger.warning("MOODLE_DB_PASSWORD not set — annotation queries will fail")

        self._langid = translation_service.load_langid()
        try:
            self._translation_llm = translation_service.build_translation_llm(config_manager)
        except Exception as e:
            logger.error(f"Failed to initialize translation LLM: {e} — ingestion translation disabled")
            self._translation_llm = None

    def get_connection(self) -> pymysql.connections.Connection:
        """Get a MariaDB connection with dict-row cursors."""
        return pymysql.connect(
            host=self.db_host,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            cursorclass=pymysql.cursors.DictCursor,
        )

    @staticmethod
    def _to_iso(unix_ts: Optional[int]) -> str:
        return datetime.fromtimestamp(unix_ts).isoformat() if unix_ts else ""

    def get_completed_annotations(
        self,
        include_extended: bool = True,  # kept for interface compatibility — no extended-transcript column in this schema
    ) -> List[Dict[str, Any]]:
        """Fetch all completed annotations with video metadata."""
        query = self._BASE_QUERY + " ORDER BY a.timecreated DESC"

        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    annotations = cursor.fetchall()
            finally:
                conn.close()

            for a in annotations:
                a["annotation_created_at"] = self._to_iso(a["annotation_created_at"])
                a["annotation_updated_at"] = self._to_iso(a["annotation_updated_at"])

            logger.info(f"Retrieved {len(annotations)} completed annotations")
            return annotations

        except Exception as e:
            logger.error(f"Failed to fetch annotations: {str(e)}")
            return []

    def get_annotations_since(
        self,
        timestamp: datetime,
        include_extended: bool = True,
    ) -> List[Dict[str, Any]]:
        """Get annotations updated since a specific timestamp."""
        query = self._BASE_QUERY + " AND a.timemodified > %s ORDER BY a.timemodified ASC"

        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query, (int(timestamp.timestamp()),))
                    annotations = cursor.fetchall()
            finally:
                conn.close()

            for a in annotations:
                a["annotation_created_at"] = self._to_iso(a["annotation_created_at"])
                a["annotation_updated_at"] = self._to_iso(a["annotation_updated_at"])

            logger.info(f"Retrieved {len(annotations)} annotations since {timestamp}")
            return annotations

        except Exception as e:
            logger.error(f"Failed to fetch annotations since {timestamp}: {str(e)}")
            return []

    def annotation_to_documents(
        self,
        annotation: Dict[str, Any],
        use_extended: bool = True,  # unused — this schema has no extended-transcript field
    ) -> List[Document]:
        """
        Convert an annotation row into a LangChain Document.

        The new schema dropped the LLM-enhanced "extended transcript" concept, so
        this always produces at most one raw-transcript document per annotation.
        """
        if not annotation.get("transcription"):
            return []

        video_filename = annotation.get("video_filename") or "unknown.mp4"
        annotation_id = annotation.get("annotation_id") or 0

        page_content = annotation["transcription"]
        source_language: Optional[str] = None
        rag_config = self.config_manager.get_config().rag
        if rag_config.enable_ingestion_translation and self._translation_llm is not None:
            whisper_lang = annotation.get("language")
            if whisper_lang:
                should_translate = whisper_lang != "fr"
                lang = whisper_lang
            else:
                # Whisper didn't tag it (nullable column, pre-migration row, or
                # Whisper unsure) — detect it ourselves, same "unknown -> run
                # langid" fallback used by the query-side node.
                lang, should_translate = translation_service.decide_translation(
                    page_content, self._langid,
                    rag_config.langid_confidence_threshold, rag_config.min_langid_chars,
                )
            if should_translate:
                prompt = translation_service.build_transcript_translation_prompt(page_content, lang)
                translated = translation_service.translate_to_french(prompt, self._translation_llm)
                source_language = lang
                if translated:
                    page_content = translated
                # else: translation failed — index untranslated, but
                # source_language is still tagged for a later retry.

        metadata = {
            "annotation_id": annotation["annotation_id"],
            "video_id": annotation["video_id"],
            "video_filename": video_filename,
            "video_filepath": annotation["video_filepath"] or "",
            "start_time": float(annotation["start_time"]) if annotation["start_time"] is not None else 0.0,
            "end_time": float(annotation["end_time"]) if annotation["end_time"] is not None else 0.0,
            "duration": float(annotation["end_time"] - annotation["start_time"]) if annotation["end_time"] is not None and annotation["start_time"] is not None else 0.0,
            "audio_filepath": annotation["audio_filepath"] or "",
            "source_type": annotation["source_type"] or "unknown",
            "project_name": annotation.get("project_name") or "unknown",
            "craft": annotation.get("craft") or "",
            "task": annotation.get("task") or "",
            "annotation_created_at": annotation["annotation_created_at"] or "",
            "type": "video_annotation",
            "transcript_type": "raw",
            "source": f"{video_filename}#{annotation_id}_raw",
            # Silo fields — the new schema has no per-project cohort restriction,
            # so every annotation is open-access (matches the prior default).
            "cohort_id": -1,
            "open_access": True,
        }
        if source_language:
            metadata["source_language"] = source_language
            if page_content != annotation["transcription"]:
                metadata["original_transcription"] = annotation["transcription"]

        return [Document(page_content=page_content, metadata=metadata)]

    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotations in the database."""
        try:
            conn = self.get_connection()
            try:
                with conn.cursor() as cursor:
                    stats: Dict[str, Any] = {}

                    cursor.execute("SELECT COUNT(*) AS c FROM mdl_local_videoelicit_annotations")
                    stats["total_annotations"] = cursor.fetchone()["c"]

                    cursor.execute(
                        "SELECT COUNT(*) AS c FROM mdl_local_videoelicit_annotations "
                        "WHERE transcriptionstatus = 'completed'"
                    )
                    stats["completed_transcriptions"] = cursor.fetchone()["c"]

                    # No extended-transcript concept in this schema.
                    stats["completed_extended"] = 0

                    cursor.execute("SELECT COUNT(*) AS c FROM mdl_local_videoelicit_videos")
                    stats["total_videos"] = cursor.fetchone()["c"]

                    cursor.execute(
                        "SELECT COUNT(DISTINCT videoid) AS c FROM mdl_local_videoelicit_annotations"
                    )
                    stats["videos_with_annotations"] = cursor.fetchone()["c"]
            finally:
                conn.close()

            return stats

        except Exception as e:
            logger.error(f"Failed to get annotation stats: {str(e)}")
            return {}
