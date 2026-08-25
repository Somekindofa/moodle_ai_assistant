"""Unit tests for services.translation_service — shared translate-to-French helpers."""

from unittest.mock import MagicMock, patch

from config.settings import ConfigurationManager
from services import translation_service


# ── load_langid — real py3langid, not mocked (same rationale as
# test_language_detection.py's equivalent RAGService test: this is the one
# test that would catch a genuine API mismatch with the installed library) ──

def test_load_langid_returns_a_working_real_identifier():
    identifier = translation_service.load_langid()

    assert identifier is not None
    en_lang, en_conf = identifier.classify("How do you blow glass?")
    fr_lang, fr_conf = identifier.classify("Comment souffler le verre ?")
    assert en_lang == "en"
    assert fr_lang == "fr"
    assert 0.0 <= float(en_conf) <= 1.0
    assert 0.0 <= float(fr_conf) <= 1.0


# ── decide_translation ──

def test_decide_translation_returns_fr_false_when_langid_unavailable():
    assert translation_service.decide_translation("How do you blow glass?", None, 0.5, 12) == ("fr", False)


def test_decide_translation_returns_fr_false_for_french_text():
    langid = MagicMock()
    langid.classify.return_value = ("fr", 0.99)
    assert translation_service.decide_translation("Comment souffler le verre ?", langid, 0.5, 12) == ("fr", False)


def test_decide_translation_returns_fr_false_below_confidence_threshold():
    langid = MagicMock()
    langid.classify.return_value = ("en", 0.2)
    assert translation_service.decide_translation("ok glass work thing", langid, 0.5, 12) == ("fr", False)


def test_decide_translation_returns_fr_false_below_min_chars():
    langid = MagicMock()
    langid.classify.return_value = ("en", 0.99)
    assert translation_service.decide_translation("ok thx", langid, 0.5, 12) == ("fr", False)


def test_decide_translation_returns_lang_true_for_confident_non_french():
    langid = MagicMock()
    langid.classify.return_value = ("en", 0.95)
    assert translation_service.decide_translation("How do you blow glass?", langid, 0.5, 12) == ("en", True)


# ── extract_text ──

def test_extract_text_handles_plain_string_content():
    response = MagicMock()
    response.content = "  Comment souffler le verre ?  "
    assert translation_service.extract_text(response) == "Comment souffler le verre ?"


def test_extract_text_handles_list_content():
    response = MagicMock()
    response.content = ["Comment ", "souffler le verre ?"]
    assert translation_service.extract_text(response) == "Comment  souffler le verre ?"


def test_extract_text_handles_other_content_types():
    response = MagicMock()
    response.content = 12345
    assert translation_service.extract_text(response) == "12345"


# ── translate_to_french ──

def test_translate_to_french_returns_translated_text():
    llm = MagicMock()
    response = MagicMock()
    response.content = "Comment souffler le verre ?"
    llm.invoke.return_value = response

    result = translation_service.translate_to_french("some prompt", llm)

    assert result == "Comment souffler le verre ?"
    llm.invoke.assert_called_once_with("some prompt")


def test_translate_to_french_returns_none_on_exception():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("API timeout")

    assert translation_service.translate_to_french("some prompt", llm) is None


def test_translate_to_french_default_does_not_retry():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("Error code: 429 - rate_limit_exceeded")

    assert translation_service.translate_to_french("some prompt", llm) is None
    llm.invoke.assert_called_once()


def test_translate_to_french_retries_on_rate_limit_and_succeeds():
    llm = MagicMock()
    success = MagicMock()
    success.content = "Comment souffler le verre ?"
    llm.invoke.side_effect = [
        Exception("Error code: 429 - rate_limit_exceeded"),
        Exception("Error code: 429 - rate_limit_exceeded"),
        success,
    ]

    with patch("services.translation_service.time.sleep") as mock_sleep:
        result = translation_service.translate_to_french("some prompt", llm, max_retries=3)

    assert result == "Comment souffler le verre ?"
    assert llm.invoke.call_count == 3
    assert mock_sleep.call_count == 2


def test_translate_to_french_gives_up_after_max_retries_on_persistent_rate_limit():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("Error code: 429 - rate_limit_exceeded")

    with patch("services.translation_service.time.sleep"):
        result = translation_service.translate_to_french("some prompt", llm, max_retries=3)

    assert result is None
    assert llm.invoke.call_count == 4  # initial attempt + 3 retries


def test_translate_to_french_does_not_retry_non_rate_limit_errors_even_with_retries_allowed():
    llm = MagicMock()
    llm.invoke.side_effect = Exception("API timeout")

    with patch("services.translation_service.time.sleep") as mock_sleep:
        result = translation_service.translate_to_french("some prompt", llm, max_retries=3)

    assert result is None
    llm.invoke.assert_called_once()
    mock_sleep.assert_not_called()


def test_translate_to_french_returns_none_on_empty_response():
    llm = MagicMock()
    response = MagicMock()
    response.content = "   "
    llm.invoke.return_value = response

    assert translation_service.translate_to_french("some prompt", llm) is None


# ── is_degenerate_text ──

def test_is_degenerate_text_flags_dot_leader_ocr_garbage():
    # Real example from a scanned consent-form PDF that hung the backfill —
    # form blank-lines extracted as walls of ". . . . . . . ." repeated
    # hundreds of times, not real prose.
    text = "Nom  " + ". . " * 200 + "MINES ParisTech"
    assert translation_service.is_degenerate_text(text) is True


def test_is_degenerate_text_does_not_flag_normal_prose():
    text = (
        "D'abord, je chauffe le verre dans le four jusqu'à ce qu'il devienne "
        "orange incandescent, puis je le façonne sur la table de marbre."
    )
    assert translation_service.is_degenerate_text(text) is False


def test_is_degenerate_text_does_not_flag_short_normal_text():
    assert translation_service.is_degenerate_text("Wear goggles at all times") is False


def test_is_degenerate_text_handles_empty_string():
    assert translation_service.is_degenerate_text("") is False


# ── build_translation_llm ──

def test_build_translation_llm_uses_zero_temperature_and_no_streaming():
    cm = ConfigurationManager()

    with patch("services.translation_service.ChatOpenAI") as MockChatOpenAI:
        translation_service.build_translation_llm(cm)

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["temperature"] == 0
        assert kwargs["streaming"] is False
        assert kwargs["model"] == cm.get_config().rag.llm_model


def test_build_translation_llm_sets_a_request_timeout():
    # A hung request with no timeout blocks forever — worse, it never even
    # raises, so translate_to_french's own retry-with-backoff never gets a
    # chance to fire. Confirmed against a real overnight backfill run that
    # hung for 30+ minutes on a single stalled call with zero timeout set.
    cm = ConfigurationManager()

    with patch("services.translation_service.ChatOpenAI") as MockChatOpenAI:
        translation_service.build_translation_llm(cm)

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["request_timeout"] is not None
        assert 0 < kwargs["request_timeout"] <= 120


def test_build_translation_llm_caps_max_tokens():
    # Defense-in-depth alongside is_degenerate_text: even if pathological
    # input slips through, a capped response length bounds how long a
    # single generation can run for.
    cm = ConfigurationManager()

    with patch("services.translation_service.ChatOpenAI") as MockChatOpenAI:
        translation_service.build_translation_llm(cm)

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["max_tokens"] is not None
        assert 0 < kwargs["max_tokens"] <= 2000


def test_build_translation_llm_disables_the_sdks_own_retry_layer():
    # max_retries is owned entirely by translate_to_french's explicit,
    # observable retry loop — a second, invisible retry layer inside the
    # SDK just makes total wait time unpredictable and compounds with ours.
    cm = ConfigurationManager()

    with patch("services.translation_service.ChatOpenAI") as MockChatOpenAI:
        translation_service.build_translation_llm(cm)

        _, kwargs = MockChatOpenAI.call_args
        assert kwargs["max_retries"] == 0


# ── prompt builders ──

def test_build_query_translation_prompt_matches_existing_query_prompt():
    prompt = translation_service.build_query_translation_prompt("How do you blow glass?", "en")

    assert "How do you blow glass?" in prompt
    assert "(en)" in prompt
    assert "UNIQUEMENT la traduction française" in prompt


def test_build_transcript_translation_prompt_includes_source_text_and_language():
    prompt = translation_service.build_transcript_translation_prompt("I heat the glass first", "en")

    assert "I heat the glass first" in prompt
    assert "(en)" in prompt
    assert "UNIQUEMENT la traduction française" in prompt


def test_build_chunk_translation_prompt_includes_source_text_and_language():
    prompt = translation_service.build_chunk_translation_prompt("Safety > Wear goggles at all times", "en")

    assert "Safety > Wear goggles at all times" in prompt
    assert "(en)" in prompt
    assert "UNIQUEMENT la traduction française" in prompt
