"""Per-user access scope resolver — queries Moodle DB for cohort membership
and course enrolments. Results are cached in-memory for 60 seconds."""

import logging
import os
import time
from typing import Optional

import pymysql
import pymysql.cursors

logger = logging.getLogger(__name__)

_COHORT_QUERY = """
    SELECT DISTINCT cm.cohortid
    FROM mdl_cohort_members cm
    WHERE cm.userid = %s
"""

_ENROL_QUERY = """
    SELECT DISTINCT e.courseid
    FROM mdl_user_enrolments ue
    JOIN mdl_enrol e ON e.id = ue.enrolid
    WHERE ue.userid = %s
      AND ue.status = 0
      AND e.status = 0
"""

_CATEGORY_COURSES_QUERY = """
    SELECT id
    FROM mdl_course
    WHERE category = %s
"""


class SiloService:
    """Resolves per-user access scope from the Moodle MySQL DB.

    Both methods cache their results for ``_cache_ttl`` seconds (default 60)
    keyed by user_id.  Raises on DB failure — callers must treat that as 503.
    """

    def __init__(
        self,
        db_host: str = "localhost",
        db_user: str = "moodleuser",
        db_password: Optional[str] = None,
        db_name: str = "moodle",
        cache_ttl: float = 60.0,
    ):
        self._db_host = db_host
        self._db_user = db_user
        self._db_password = db_password or os.getenv("MOODLE_DB_PASSWORD", "")
        self._db_name = db_name
        self._cache_ttl = cache_ttl
        self._cohort_cache: dict[int, tuple[list[int], float]] = {}
        self._course_cache: dict[int, tuple[list[str], float]] = {}
        self._category_course_cache: dict[int, tuple[list[str], float]] = {}

    def _connect(self):
        return pymysql.connect(
            host=self._db_host,
            user=self._db_user,
            password=self._db_password,
            database=self._db_name,
            cursorclass=pymysql.cursors.Cursor,
            connect_timeout=5,
        )

    def get_allowed_cohorts(self, user_id: int) -> list[int]:
        """Return Moodle cohort IDs the user belongs to."""
        cached, ts = self._cohort_cache.get(user_id, (None, 0.0))
        if cached is not None and (time.time() - ts) < self._cache_ttl:
            return list(cached)

        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_COHORT_QUERY, (user_id,))
                rows = cur.fetchall()
        finally:
            if conn is not None:
                conn.close()

        result = [row[0] for row in rows]
        self._cohort_cache[user_id] = (result, time.time())
        logger.debug(f"SiloService: user {user_id} cohorts={result}")
        return result

    def get_enrolled_course_ids(self, user_id: int) -> list[str]:
        """Return active Moodle course IDs the user is enrolled in."""
        cached, ts = self._course_cache.get(user_id, (None, 0.0))
        if cached is not None and (time.time() - ts) < self._cache_ttl:
            return list(cached)

        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_ENROL_QUERY, (user_id,))
                rows = cur.fetchall()
        finally:
            if conn is not None:
                conn.close()

        result = [str(row[0]) for row in rows]
        self._course_cache[user_id] = (result, time.time())
        logger.debug(f"SiloService: user {user_id} courses={result}")
        return result

    def get_course_ids_by_category(self, category_id: int) -> list[str]:
        """Return Moodle course IDs belonging to the given course category.

        Used to narrow retrieval to a student's selected craft domain (see
        DOMAIN_MAP in services/rag_service.py) — independent of enrolment.
        """
        cached, ts = self._category_course_cache.get(category_id, (None, 0.0))
        if cached is not None and (time.time() - ts) < self._cache_ttl:
            return list(cached)

        conn = None
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CATEGORY_COURSES_QUERY, (category_id,))
                rows = cur.fetchall()
        finally:
            if conn is not None:
                conn.close()

        result = [str(row[0]) for row in rows]
        self._category_course_cache[category_id] = (result, time.time())
        logger.debug(f"SiloService: category {category_id} courses={result}")
        return result
