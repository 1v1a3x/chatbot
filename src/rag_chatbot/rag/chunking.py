"""Document chunking strategies for RAG."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """A document chunk with metadata."""

    text: str
    index: int
    metadata: Dict[str, Any]
    start_char: int
    end_char: int


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Split text into chunks.

        Args:
            text: Document text to chunk
            metadata: Metadata to attach to chunks

        Returns:
            List of chunks
        """
        pass


class RecursiveChunker(ChunkingStrategy):
    """Recursive text chunking with overlap.

    Splits text recursively using separators (paragraphs, sentences, words)
    while respecting chunk size limits.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None,
    ):
        """Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            separators: List of separators in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Recursively chunk text."""
        metadata = metadata or {}
        chunks = []

        # Start with entire text
        texts = [text]

        # Recursively split with each separator
        for separator in self.separators:
            new_texts = []
            for t in texts:
                if len(t) > self.chunk_size:
                    # Split and keep separator
                    parts = t.split(separator)
                    for i, part in enumerate(parts):
                        if i < len(parts) - 1 and separator.strip():
                            part += separator.strip()
                        if part.strip():
                            new_texts.append(part)
                else:
                    new_texts.append(t)
            texts = new_texts

        # Merge small chunks
        chunks_texts = self._merge_chunks(texts)

        # Create chunks with overlap
        char_idx = 0
        for i, chunk_text in enumerate(chunks_texts):
            # Find start position in original text
            start_char = text.find(chunk_text, max(0, char_idx - self.chunk_overlap))
            end_char = start_char + len(chunk_text)

            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=i,
                    metadata={**metadata, "chunk_index": i},
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            char_idx = end_char - self.chunk_overlap

        return chunks

    def _merge_chunks(self, texts: List[str]) -> List[str]:
        """Merge small chunks to reach target size."""
        merged = []
        current = ""

        for text in texts:
            if len(current) + len(text) <= self.chunk_size:
                current += text
            else:
                if current:
                    merged.append(current.strip())
                current = text

        if current:
            merged.append(current.strip())

        return merged


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on sentence embeddings similarity.

    Groups sentences into chunks based on semantic similarity.
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        similarity_threshold: float = 0.7,
    ):
        """Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum chunk size
            similarity_threshold: Similarity threshold for grouping
        """
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk text based on semantic similarity."""
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        char_idx = 0

        for i in range(1, len(sentences)):
            # Simple heuristic: check sentence similarity based on word overlap
            # In production, use embeddings
            similarity = self._sentence_similarity(" ".join(current_chunk), sentences[i])

            if (
                similarity >= self.similarity_threshold
                and len(" ".join(current_chunk + [sentences[i]])) <= self.max_chunk_size
            ):
                current_chunk.append(sentences[i])
            else:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        metadata={**(metadata or {}), "chunk_index": len(chunks)},
                        start_char=char_idx,
                        end_char=char_idx + len(chunk_text),
                    )
                )
                char_idx += len(chunk_text)
                current_chunk = [sentences[i]]

        # Don't forget last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    metadata={**(metadata or {}), "chunk_index": len(chunks)},
                    start_char=char_idx,
                    end_char=char_idx + len(chunk_text),
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate simple word overlap similarity."""
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


class MarkdownChunker(ChunkingStrategy):
    """Chunk Markdown documents preserving structure.

    Chunks by headers while respecting size limits.
    """

    def __init__(self, chunk_size: int = 1000):
        """Initialize Markdown chunker.

        Args:
            chunk_size: Maximum chunk size
        """
        self.chunk_size = chunk_size

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """Chunk Markdown by headers."""
        metadata = metadata or {}
        chunks = []

        # Split by headers
        header_pattern = r"(^|\n)(#{1,6}\s+.+?)(?=\n#{1,6}\s|$)"
        sections = re.split(header_pattern, text, flags=re.MULTILINE)

        current_chunk = ""
        current_headers = []
        char_idx = 0
        chunk_idx = 0

        for section in sections:
            if not section.strip():
                continue

            # Check if it's a header
            if section.strip().startswith("#"):
                current_headers = [section.strip()]
                continue

            # Add content
            content = "\n".join(current_headers + [section])

            if len(current_chunk) + len(content) > self.chunk_size:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            index=chunk_idx,
                            metadata={
                                **metadata,
                                "chunk_index": chunk_idx,
                                "headers": current_headers,
                            },
                            start_char=char_idx,
                            end_char=char_idx + len(current_chunk),
                        )
                    )
                    char_idx += len(current_chunk)
                    chunk_idx += 1
                current_chunk = content
            else:
                current_chunk += "\n\n" + content

        # Don't forget last chunk
        if current_chunk:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    index=chunk_idx,
                    metadata={**metadata, "chunk_index": chunk_idx, "headers": current_headers},
                    start_char=char_idx,
                    end_char=char_idx + len(current_chunk),
                )
            )

        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> ChunkingStrategy:
    """Get chunker by strategy name.

    Args:
        strategy: Chunking strategy name (recursive, semantic, markdown)
        **kwargs: Additional arguments for chunker

    Returns:
        Chunking strategy instance
    """
    strategies = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "markdown": MarkdownChunker,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(strategies.keys())}")

    return strategies[strategy](**kwargs)
