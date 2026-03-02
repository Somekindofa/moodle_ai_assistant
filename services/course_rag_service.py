"""Course RAG service — per-course ChromaDB collections with semantic chunking."""

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Semantic chunker
# ─────────────────────────────────────────────────────────────────

HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TARGET_TOKENS = 400   # ~1600 chars
MIN_TOKENS = 50       # ~200 chars — merge if below this


def _approx_tokens(text: str) -> int:
    """Rough token estimate: 4 chars ≈ 1 token."""
    return max(1, len(text) // 4)


class SemanticChunker:
    """Splits structured documents (HTML / PDF / DOCX) into semantic chunks.

    Each chunk captures a coherent section bounded by headings and paragraph
    breaks.  The heading breadcrumb is prepended to the chunk text so that
    vector search is hierarchy-aware.
    """

    # ── HTML ──────────────────────────────────────────────────────

    def chunk_html(
        self,
        html: str,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Parse HTML and return semantic Document chunks."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed — cannot chunk HTML")
            return []

        soup = BeautifulSoup(html, "html.parser")
        return self._walk_soup(soup, base_metadata)

    def _walk_soup(
        self, soup: Any, base_metadata: Dict[str, Any]
    ) -> List[Document]:
        from bs4 import NavigableString, Tag

        heading_stack: List[str] = []   # breadcrumb of current headings
        buffer: List[str] = []           # accumulated paragraph text
        chunks: List[Document] = []

        def flush(force: bool = False) -> None:
            text = " ".join(buffer).strip()
            if not text:
                buffer.clear()
                return
            if not force and _approx_tokens(text) < MIN_TOKENS:
                return  # too small — keep accumulating
            _emit(text)
            buffer.clear()

        def _emit(text: str) -> None:
            heading_path = " > ".join(heading_stack) if heading_stack else ""
            # Prepend breadcrumb so the embedding encodes positional context
            full_text = f"{heading_path}\n\n{text}".strip() if heading_path else text
            idx = len(chunks)
            meta = {
                **base_metadata,
                "chunk_index": idx,
                "heading_path": heading_path,
                "source": f"course_{base_metadata.get('course_id')}_module_{base_metadata.get('module_id')}_chunk_{idx}",
            }
            chunks.append(Document(page_content=full_text, metadata=meta))

        for element in soup.descendants:
            if not isinstance(element, (NavigableString,)) and not isinstance(element, type(soup)):
                tag_name = getattr(element, "name", None)
                if tag_name is None:
                    continue

                if tag_name in HEADING_TAGS:
                    flush(force=True)
                    level = int(tag_name[1])
                    heading_text = element.get_text(" ", strip=True)
                    # Keep heading_stack up to (level - 1) then add new heading
                    heading_stack[:] = heading_stack[: level - 1]
                    heading_stack.append(heading_text)

                elif tag_name in ("p", "li", "td", "th", "dd", "dt", "blockquote"):
                    text = element.get_text(" ", strip=True)
                    if text:
                        buffer.append(text)
                        # Flush if buffer is large enough
                        combined = " ".join(buffer)
                        if _approx_tokens(combined) >= TARGET_TOKENS:
                            flush(force=True)

        flush(force=True)
        return chunks

    # ── PDF ───────────────────────────────────────────────────────

    def chunk_pdf(
        self,
        file_bytes: bytes,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Extract text from PDF and return semantic Document chunks."""
        try:
            import fitz  # pymupdf
        except ImportError:
            logger.error("pymupdf not installed — cannot chunk PDF")
            return []

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return []

        heading_stack: List[str] = []
        buffer: List[str] = []
        chunks: List[Document] = []
        page_images: Dict[int, int] = {}   # page_num → image count

        def flush(force: bool = False) -> None:
            text = " ".join(buffer).strip()
            if not text:
                buffer.clear()
                return
            if not force and _approx_tokens(text) < MIN_TOKENS:
                return
            heading_path = " > ".join(heading_stack) if heading_stack else ""
            full_text = f"{heading_path}\n\n{text}".strip() if heading_path else text
            idx = len(chunks)
            meta = {
                **base_metadata,
                "chunk_index": idx,
                "heading_path": heading_path,
                "source": f"course_{base_metadata.get('course_id')}_module_{base_metadata.get('module_id')}_chunk_{idx}",
            }
            if page_images:
                meta["image_pages"] = ",".join(str(p) for p in sorted(page_images))
            chunks.append(Document(page_content=full_text, metadata=meta))
            buffer.clear()
            page_images.clear()

        # Heuristic: the median font size of body text
        all_sizes: List[float] = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            all_sizes.append(span.get("size", 12))

        if all_sizes:
            all_sizes.sort()
            body_size = all_sizes[len(all_sizes) // 2]  # median
        else:
            body_size = 12.0

        heading_threshold = body_size * 1.15  # 15% larger than body = heading

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                btype = block.get("type")
                if btype == 1:  # image block
                    page_images[page_num] = page_images.get(page_num, 0) + 1
                    continue
                if btype != 0:
                    continue

                for line in block.get("lines", []):
                    line_text = " ".join(
                        span["text"] for span in line.get("spans", [])
                    ).strip()
                    if not line_text:
                        continue

                    # Check if this line looks like a heading
                    span_sizes = [s.get("size", 12) for s in line.get("spans", [])]
                    avg_size = sum(span_sizes) / len(span_sizes) if span_sizes else 12
                    is_bold = any(
                        s.get("flags", 0) & 0b10000 for s in line.get("spans", [])
                    )

                    if avg_size >= heading_threshold or (is_bold and len(line_text) < 80):
                        flush(force=True)
                        # Approximate heading level from font size
                        level = max(1, min(3, round((heading_threshold - avg_size + heading_threshold) / heading_threshold)))
                        heading_stack[:] = heading_stack[:level - 1]
                        heading_stack.append(line_text)
                    else:
                        buffer.append(line_text)
                        combined = " ".join(buffer)
                        if _approx_tokens(combined) >= TARGET_TOKENS:
                            flush(force=True)

        flush(force=True)
        return chunks

    # ── DOCX ──────────────────────────────────────────────────────

    def chunk_docx(
        self,
        file_bytes: bytes,
        base_metadata: Dict[str, Any],
    ) -> List[Document]:
        """Extract text from DOCX and return semantic Document chunks."""
        try:
            import docx as python_docx
        except ImportError:
            logger.error("python-docx not installed — cannot chunk DOCX")
            return []

        try:
            doc = python_docx.Document(io.BytesIO(file_bytes))
        except Exception as e:
            logger.error(f"Failed to open DOCX: {e}")
            return []

        heading_stack: List[str] = []
        buffer: List[str] = []
        chunks: List[Document] = []

        def flush(force: bool = False) -> None:
            text = " ".join(buffer).strip()
            if not text:
                buffer.clear()
                return
            if not force and _approx_tokens(text) < MIN_TOKENS:
                return
            heading_path = " > ".join(heading_stack) if heading_stack else ""
            full_text = f"{heading_path}\n\n{text}".strip() if heading_path else text
            idx = len(chunks)
            meta = {
                **base_metadata,
                "chunk_index": idx,
                "heading_path": heading_path,
                "source": f"course_{base_metadata.get('course_id')}_module_{base_metadata.get('module_id')}_chunk_{idx}",
            }
            chunks.append(Document(page_content=full_text, metadata=meta))
            buffer.clear()

        # Process paragraphs
        for para in doc.paragraphs:
            style_name = para.style.name if para.style else ""
            text = para.text.strip()
            if not text:
                continue

            if style_name.startswith("Heading"):
                flush(force=True)
                # Extract level from style name e.g. "Heading 1" → 1
                m = re.search(r"(\d+)", style_name)
                level = int(m.group(1)) if m else 1
                heading_stack[:] = heading_stack[:level - 1]
                heading_stack.append(text)
            else:
                buffer.append(text)
                combined = " ".join(buffer)
                if _approx_tokens(combined) >= TARGET_TOKENS:
                    flush(force=True)

        # Process tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    buffer.append(f"Row: {row_text}")
                    combined = " ".join(buffer)
                    if _approx_tokens(combined) >= TARGET_TOKENS:
                        flush(force=True)

        flush(force=True)
        return chunks


# ─────────────────────────────────────────────────────────────────
# CourseRAGService
# ─────────────────────────────────────────────────────────────────

class CourseRAGService:
    """Manages per-course ChromaDB collections for Moodle course content.

    Each course gets its own Chroma collection named ``course_{course_id}``.
    The shared embeddings model (same as RAGService) is injected at construction
    time to avoid loading the model twice.
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        persist_directory: str,
    ) -> None:
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.chunker = SemanticChunker()
        # Cache open Chroma collection handles keyed by course_id string
        self._collections: Dict[str, Chroma] = {}
        logger.info("CourseRAGService initialized")

    # ── Collection management ─────────────────────────────────────

    def _collection_name(self, course_id: str) -> str:
        return f"course_{course_id}"

    def _get_collection(self, course_id: str) -> Chroma:
        """Return (or lazily open) the Chroma collection for a course."""
        if course_id not in self._collections:
            self._collections[course_id] = Chroma(
                collection_name=self._collection_name(course_id),
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(f"Opened ChromaDB collection for course {course_id}")
        return self._collections[course_id]

    def delete_collection(self, course_id: str) -> None:
        """Drop the entire ChromaDB collection for a course."""
        name = self._collection_name(course_id)
        try:
            # Get the underlying chromadb client from the langchain wrapper
            store = self._get_collection(course_id)
            store._client.delete_collection(name)
            # Remove from cache
            self._collections.pop(course_id, None)
            logger.info(f"Deleted ChromaDB collection '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{name}': {e}")
            raise

    # ── Ingestion ─────────────────────────────────────────────────

    def ingest_module(
        self,
        course_id: str,
        module_id: str,
        module_type: str,
        module_name: str,
        section_name: str,
        content_html: Optional[str] = None,
        content_raw_b64: Optional[str] = None,
        file_extension: Optional[str] = None,
    ) -> int:
        """Chunk and embed a course module into the per-course collection.

        Returns the number of chunks added.
        """
        base_metadata = {
            "type": "course_content",
            "course_id": course_id,
            "module_id": module_id,
            "module_type": module_type,
            "module_name": module_name,
            "section_name": section_name,
        }

        chunks: List[Document] = []

        if content_html:
            chunks = self.chunker.chunk_html(content_html, base_metadata)
        elif content_raw_b64 and file_extension:
            file_bytes = base64.b64decode(content_raw_b64)
            ext = file_extension.lower().lstrip(".")
            if ext == "pdf":
                chunks = self.chunker.chunk_pdf(file_bytes, base_metadata)
            elif ext in ("docx", "doc"):
                chunks = self.chunker.chunk_docx(file_bytes, base_metadata)
            else:
                logger.warning(f"Unsupported file extension '{ext}' — skipping")
                return 0
        else:
            logger.warning(f"No content provided for module {module_id} — skipping")
            return 0

        if not chunks:
            logger.info(f"No chunks produced for module {module_id}")
            return 0

        collection = self._get_collection(course_id)
        # Infomaniak embedding API accepts at most 99 items per call.
        batch_size = 99
        for i in range(0, len(chunks), batch_size):
            collection.add_documents(chunks[i : i + batch_size])
        logger.info(
            f"Indexed {len(chunks)} chunks for course {course_id} / module {module_id}"
        )
        return len(chunks)

    def delete_module(self, course_id: str, module_id: str) -> int:
        """Remove all chunks belonging to a module from the course collection.

        Returns the number of chunks deleted.
        """
        try:
            collection = self._get_collection(course_id)
            results = collection.get(where={"module_id": module_id})
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(
                    f"Deleted {len(ids)} chunks for course {course_id} / module {module_id}"
                )
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to delete module {module_id} from course {course_id}: {e}")
            raise

    # ── Retrieval ─────────────────────────────────────────────────

    def similarity_search(
        self, query: str, course_id: str, k: int = 5
    ) -> List[Document]:
        """MMR search within a single course's collection."""
        try:
            collection = self._get_collection(course_id)
            # Check if collection has any documents
            data = collection.get()
            if not data.get("ids"):
                logger.info(f"Course collection '{course_id}' is empty")
                return []

            results = collection.max_marginal_relevance_search(query, k=k)

            # Deduplicate by source
            seen: set = set()
            unique: List[Document] = []
            for doc in results:
                src = doc.metadata.get("source", "")
                if src not in seen:
                    seen.add(src)
                    unique.append(doc)

            logger.info(
                f"Course search for '{course_id}' returned {len(unique)} results"
            )
            return unique

        except Exception as e:
            logger.error(f"Course similarity search failed for course {course_id}: {e}")
            return []

    # ── Status ────────────────────────────────────────────────────

    def get_course_status(self, course_id: str) -> Dict[str, Any]:
        """Return chunk and module counts for a course collection."""
        try:
            collection = self._get_collection(course_id)
            data = collection.get()
            ids = data.get("ids", [])
            metadatas = data.get("metadatas", [])
            module_ids = {m.get("module_id") for m in metadatas if m.get("module_id")}
            return {
                "course_id": course_id,
                "collection": self._collection_name(course_id),
                "chunk_count": len(ids),
                "module_count": len(module_ids),
            }
        except Exception as e:
            logger.error(f"Failed to get status for course {course_id}: {e}")
            return {
                "course_id": course_id,
                "collection": self._collection_name(course_id),
                "chunk_count": 0,
                "module_count": 0,
            }
