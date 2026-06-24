"""Unit tests for SiloService — all DB calls are mocked."""

import time
import pytest
from unittest.mock import MagicMock, patch


def _make_service():
    from services.silo_service import SiloService
    return SiloService(db_password="test")


def _mock_cursor(rows):
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    return cursor


def _mock_conn(cursor):
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn


# ── get_allowed_cohorts ──────────────────────────────────────────────────────

def test_get_allowed_cohorts_returns_ids():
    svc = _make_service()
    cursor = _mock_cursor([(7,), (42,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_allowed_cohorts(99)
    assert result == [7, 42]


def test_get_allowed_cohorts_empty_for_unknown_user():
    svc = _make_service()
    cursor = _mock_cursor([])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_allowed_cohorts(0)
    assert result == []


def test_get_allowed_cohorts_cached_on_second_call():
    svc = _make_service()
    cursor = _mock_cursor([(1,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_allowed_cohorts(5)
        svc.get_allowed_cohorts(5)   # should use cache
    assert mock_connect.call_count == 1


def test_get_allowed_cohorts_cache_expires():
    svc = _make_service()
    svc._cache_ttl = 0.05   # 50 ms for test speed
    cursor = _mock_cursor([(1,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_allowed_cohorts(5)
        time.sleep(0.1)
        svc.get_allowed_cohorts(5)   # cache expired
    assert mock_connect.call_count == 2


# ── get_enrolled_course_ids ──────────────────────────────────────────────────

def test_get_enrolled_course_ids_returns_string_ids():
    svc = _make_service()
    cursor = _mock_cursor([(10,), (23,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_enrolled_course_ids(99)
    assert result == ["10", "23"]


def test_get_enrolled_course_ids_empty_when_not_enrolled():
    svc = _make_service()
    cursor = _mock_cursor([])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn):
        result = svc.get_enrolled_course_ids(1)
    assert result == []


def test_get_enrolled_course_ids_cached():
    svc = _make_service()
    cursor = _mock_cursor([(10,)])
    conn = _mock_conn(cursor)
    with patch("services.silo_service.pymysql.connect", return_value=conn) as mock_connect:
        svc.get_enrolled_course_ids(5)
        svc.get_enrolled_course_ids(5)
    assert mock_connect.call_count == 1


# ── DB failure ───────────────────────────────────────────────────────────────

def test_get_allowed_cohorts_raises_on_db_error():
    svc = _make_service()
    with patch("services.silo_service.pymysql.connect", side_effect=Exception("DB down")):
        with pytest.raises(Exception, match="DB down"):
            svc.get_allowed_cohorts(1)


def test_get_enrolled_course_ids_raises_on_db_error():
    svc = _make_service()
    with patch("services.silo_service.pymysql.connect", side_effect=Exception("DB down")):
        with pytest.raises(Exception, match="DB down"):
            svc.get_enrolled_course_ids(1)
