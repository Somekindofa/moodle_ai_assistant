"""Service for managing video annotation database operations."""

import logging
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from langchain_core.documents.base import Document

from config.settings import ConfigurationManager


logger = logging.getLogger(__name__)


class AnnotationService:
    """Service for reading and managing video annotations from SQLite."""

    def __init__(
        self, 
        config_manager: ConfigurationManager,
        db_path: str = "chroma_langchain_db/elicitations_db/annotations.db"
    ):
        self.config_manager = config_manager
        self.db_path = db_path
        self._ensure_database_exists()

    def _ensure_database_exists(self) -> None:
        """Check if database exists and is accessible."""
        if not Path(self.db_path).exists():
            logger.warning(f"Annotation database not found at {self.db_path}")
        else:
            logger.info(f"Connected to annotation database at {self.db_path}")

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def get_completed_annotations(
        self, 
        include_extended: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Fetch all completed annotations with video metadata.
        
        Args:
            include_extended: Whether to include extended transcripts
            
        Returns:
            List of annotation dictionaries with video metadata
        """
        query = """
        SELECT 
            a.id as annotation_id,
            a.video_id,
            a.start_time,
            a.end_time,
            a.audio_filename,
            a.audio_filepath,
            a.transcription,
            a.transcription_status,
            a.extended_transcript,
            a.extended_transcript_status,
            a.feedback,
            a.created_at as annotation_created_at,
            a.updated_at as annotation_updated_at,
            v.filename as video_filename,
            v.filepath as video_filepath,
            v.duration as video_duration,
            v.source_type,
            v.batch_position,
            p.name as project_name,
            p.description as project_description
        FROM annotations a
        JOIN videos v ON a.video_id = v.id
        LEFT JOIN projects p ON v.project_id = p.id
        WHERE a.transcription_status = 'completed'
        """
        
        if include_extended:
            query += " AND a.extended_transcript_status = 'completed'"
        
        query += " ORDER BY a.created_at DESC"
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            
            annotations = []
            for row in cursor.fetchall():
                annotations.append(dict(row))
            
            conn.close()
            logger.info(f"Retrieved {len(annotations)} completed annotations")
            return annotations
            
        except Exception as e:
            logger.error(f"Failed to fetch annotations: {str(e)}")
            return []

    def get_annotations_since(
        self, 
        timestamp: datetime,
        include_extended: bool = True
    ) -> List[Dict[str, Any]]:
        """Get annotations updated since a specific timestamp."""
        query = """
        SELECT 
            a.id as annotation_id,
            a.video_id,
            a.start_time,
            a.end_time,
            a.audio_filename,
            a.audio_filepath,
            a.transcription,
            a.transcription_status,
            a.extended_transcript,
            a.extended_transcript_status,
            a.feedback,
            a.created_at as annotation_created_at,
            a.updated_at as annotation_updated_at,
            v.filename as video_filename,
            v.filepath as video_filepath,
            v.duration as video_duration,
            v.source_type,
            v.batch_position,
            p.name as project_name,
            p.description as project_description
        FROM annotations a
        JOIN videos v ON a.video_id = v.id
        LEFT JOIN projects p ON v.project_id = p.id
        WHERE a.transcription_status = 'completed'
        AND a.updated_at > ?
        """
        
        if include_extended:
            query += " AND a.extended_transcript_status = 'completed'"
        
        query += " ORDER BY a.updated_at ASC"
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query, (timestamp.isoformat(),))
            
            annotations = []
            for row in cursor.fetchall():
                annotations.append(dict(row))
            
            conn.close()
            logger.info(f"Retrieved {len(annotations)} annotations since {timestamp}")
            return annotations
            
        except Exception as e:
            logger.error(f"Failed to fetch annotations since {timestamp}: {str(e)}")
            return []

    def annotation_to_documents(
        self,
        annotation: Dict[str, Any],
        use_extended: bool = True
    ) -> List[Document]:
        """
        Convert annotation to LangChain Document objects.
        
        Creates separate documents for transcription and extended transcript if available.
        
        Args:
            annotation: Annotation dictionary from database
            use_extended: Whether to create document for extended transcript
            
        Returns:
            List of Document objects (1-2 documents per annotation)
        """
        documents = []
        
        # Base metadata shared by both documents
        base_metadata = {
            "annotation_id": annotation["annotation_id"],
            "video_id": annotation["video_id"],
            "video_filename": annotation["video_filename"],
            "video_filepath": annotation["video_filepath"],
            "start_time": annotation["start_time"],
            "end_time": annotation["end_time"],
            "duration": annotation["end_time"] - annotation["start_time"],
            "audio_filepath": annotation["audio_filepath"],
            "source_type": annotation["source_type"],
            "project_name": annotation.get("project_name", "unknown"),
            "annotation_created_at": annotation["annotation_created_at"],
            "type": "video_annotation"
        }
        
        # Document 1: Raw transcription
        if annotation.get("transcription"):
            transcription_metadata = base_metadata.copy()
            transcription_metadata["transcript_type"] = "raw"
            transcription_metadata["source"] = f"{annotation['video_filename']}#{annotation['annotation_id']}_raw"
            
            documents.append(Document(
                page_content=annotation["transcription"],
                metadata=transcription_metadata
            ))
        
        # Document 2: Extended transcript (LLM-enhanced)
        if use_extended and annotation.get("extended_transcript"):
            extended_metadata = base_metadata.copy()
            extended_metadata["transcript_type"] = "extended"
            extended_metadata["source"] = f"{annotation['video_filename']}#{annotation['annotation_id']}_extended"
            
            documents.append(Document(
                page_content=annotation["extended_transcript"],
                metadata=extended_metadata
            ))
        
        return documents

    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotations in database."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            # Total annotations
            cursor.execute("SELECT COUNT(*) FROM annotations")
            stats["total_annotations"] = cursor.fetchone()[0]
            
            # Completed transcriptions
            cursor.execute(
                "SELECT COUNT(*) FROM annotations WHERE transcription_status = 'completed'"
            )
            stats["completed_transcriptions"] = cursor.fetchone()[0]
            
            # Completed extended transcripts
            cursor.execute(
                "SELECT COUNT(*) FROM annotations WHERE extended_transcript_status = 'completed'"
            )
            stats["completed_extended"] = cursor.fetchone()[0]
            
            # Total videos
            cursor.execute("SELECT COUNT(*) FROM videos")
            stats["total_videos"] = cursor.fetchone()[0]
            
            # Videos with annotations
            cursor.execute(
                "SELECT COUNT(DISTINCT video_id) FROM annotations"
            )
            stats["videos_with_annotations"] = cursor.fetchone()[0]
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get annotation stats: {str(e)}")
            return {}
