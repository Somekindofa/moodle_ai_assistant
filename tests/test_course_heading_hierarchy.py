"""Offline tests for SemanticChunker heading hierarchy + breadcrumb translation.

Both bugs these cover corrupt the *embedded* text, not just the display:
SemanticChunker prepends the breadcrumb to page_content before it is
embedded, so a fabricated parent (Bug A) or a per-chunk-random breadcrumb
translation (Bug B) poisons the vector itself.

No live backend, no network: the chunker is pure, and the translation LLM is
a stub whose per-call output varies on purpose.
"""

from unittest.mock import MagicMock

from langchain_core.documents.base import Document

from config.settings import ConfigurationManager
from services.course_rag_service import CourseRAGService, SemanticChunker


META = {"course_id": "109", "module_id": "42"}


def _paths(html):
    return [c.metadata["heading_path"] for c in SemanticChunker().chunk_html(html, META)]


def _body(words, n=30):
    """A paragraph long enough to clear MIN_TOKENS (50 tokens ≈ 200 chars)."""
    return "<p>" + (" ".join([words] * n)) + "</p>"


# ── Bug A: heading hierarchy ──────────────────────────────────────────

def test_page_without_h1_keeps_h2_siblings_as_siblings():
    html = (
        "<h2>Le fusokalamo</h2>" + _body("canne")
        + "<h2>Le pontil</h2>" + _body("pontil")
    )
    assert _paths(html) == ["Le fusokalamo", "Le pontil"]


def test_h1_present_nests_h2_under_it():
    html = (
        "<h1>Outils</h1>" + _body("intro")
        + "<h2>Le fusokalamo</h2>" + _body("canne")
        + "<h2>Le pontil</h2>" + _body("pontil")
    )
    assert _paths(html) == [
        "Outils",
        "Outils > Le fusokalamo",
        "Outils > Le pontil",
    ]


def test_skipped_level_does_not_invent_a_missing_parent():
    html = "<h1>Outils</h1>" + _body("intro") + "<h3>Types de cannes</h3>" + _body("types")
    assert _paths(html) == ["Outils", "Outils > Types de cannes"]


def test_deeper_then_shallower_pops_back_to_the_right_ancestor():
    html = (
        "<h1>A</h1>" + _body("a")
        + "<h2>B</h2>" + _body("b")
        + "<h3>C</h3>" + _body("c")
        + "<h2>D</h2>" + _body("d")
        + "<h1>E</h1>" + _body("e")
    )
    assert _paths(html) == ["A", "A > B", "A > B > C", "A > D", "E"]


def test_page_starting_at_h3_has_no_fabricated_root():
    html = "<h3>Recuit</h3>" + _body("recuit") + "<h3>Defauts</h3>" + _body("defauts")
    assert _paths(html) == ["Recuit", "Defauts"]


def test_breadcrumb_is_prepended_to_the_embedded_text():
    html = "<h2>Le pontil</h2>" + _body("pontil")
    doc = SemanticChunker().chunk_html(html, META)[0]
    assert doc.page_content.startswith("Le pontil\n\n")


# ── Bug B: breadcrumb translation stability ───────────────────────────

def _service():
    svc = CourseRAGService(
        embeddings=MagicMock(), persist_directory="/tmp/test_chroma", config_manager=None
    )
    svc._langid = MagicMock()
    svc._langid.classify.return_value = ("el", 0.99)
    return svc


def _rag_config():
    cm = ConfigurationManager()
    cm.get_config().rag.enable_ingestion_translation = True
    return cm.get_config().rag


class _DriftingLLM:
    """Stub LLM reproducing the observed failure: a different French title
    every time the same Greek heading is sent inside a different chunk."""

    def __init__(self):
        self.titles = ["Decoupage", "Torsade", "Tronconnage", "Affutage", "Torsion"]
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if "Titre original (" in prompt:      # heading-only prompt
            return MagicMock(content=self.titles[len(self.calls) % len(self.titles)])
        return MagicMock(content=f"corps traduit {len(self.calls)}")


def _greek_page_chunks(n=5):
    path = "Ανόπτηση, ψυχρή κατεργασία και ελαττώματα"
    return [
        Document(
            page_content=f"{path}\n\nΚείμενο παραγράφου {i} με τεχνικές λεπτομέρειες.",
            metadata={"heading_path": path, "chunk_index": i},
        )
        for i in range(n)
    ]


def test_one_breadcrumb_translation_is_reused_by_every_chunk_of_the_page():
    svc = _service()
    svc._translation_llm = _DriftingLLM()

    out = svc._translate_chunks_if_needed(_greek_page_chunks(5), _rag_config())

    roots = {d.page_content.split("\n\n")[0] for d in out}
    assert len(roots) == 1, f"breadcrumb drifted across chunks: {roots}"

    heading_calls = [p for p in svc._translation_llm.calls if "Titre original (" in p]
    assert len(heading_calls) == 1, f"expected 1 heading call, got {len(heading_calls)}"


def test_body_is_translated_without_the_breadcrumb_glued_on():
    svc = _service()
    svc._translation_llm = _DriftingLLM()

    svc._translate_chunks_if_needed(_greek_page_chunks(2), _rag_config())

    body_calls = [p for p in svc._translation_llm.calls if "Titre original (" not in p]
    assert body_calls, "no body translation happened"
    for p in body_calls:
        assert "Ανόπτηση" not in p, "breadcrumb leaked into the body translation prompt"


def test_same_heading_renders_identically_as_leaf_and_as_ancestor():
    svc = _service()
    svc._translation_llm = _DriftingLLM()

    chunks = [
        Document(page_content="Εργαλεία\n\nΕισαγωγή στα εργαλεία.",
                 metadata={"heading_path": "Εργαλεία", "chunk_index": 0}),
        Document(page_content="Εργαλεία > Το πόντιλ\n\nΤο πόντιλ χρησιμοποιείται.",
                 metadata={"heading_path": "Εργαλεία > Το πόντιλ", "chunk_index": 1}),
    ]
    out = svc._translate_chunks_if_needed(chunks, _rag_config())

    leaf = out[0].page_content.split("\n\n")[0]
    ancestor = out[1].page_content.split("\n\n")[0].split(" > ")[0]
    assert leaf == ancestor, f"{leaf!r} != {ancestor!r}"


def test_heading_path_metadata_stays_in_the_source_language():
    svc = _service()
    svc._translation_llm = _DriftingLLM()

    out = svc._translate_chunks_if_needed(_greek_page_chunks(2), _rag_config())

    for d in out:
        assert d.metadata["heading_path"] == "Ανόπτηση, ψυχρή κατεργασία και ελαττώματα"


def test_body_translation_failure_keeps_the_whole_chunk_original():
    svc = _service()
    svc._translation_llm = MagicMock()
    svc._translation_llm.invoke.side_effect = Exception("API timeout")

    chunks = _greek_page_chunks(1)
    out = svc._translate_chunks_if_needed(chunks, _rag_config())

    assert out[0].page_content == chunks[0].page_content
    assert "original_text" not in out[0].metadata


def test_chunk_without_a_baked_in_breadcrumb_is_translated_as_one_blob():
    """Guards the pre-existing contract: chunks whose page_content does not
    start with heading_path make no extra heading call."""
    svc = _service()
    svc._translation_llm = _DriftingLLM()

    chunks = [Document(page_content="Κείμενο χωρίς επικεφαλίδα στην αρχή.",
                       metadata={"heading_path": "Ασφάλεια", "chunk_index": 0})]
    out = svc._translate_chunks_if_needed(chunks, _rag_config())

    assert [p for p in svc._translation_llm.calls if "Titre original (" in p] == []
    assert out[0].page_content == "corps traduit 1"
