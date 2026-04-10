from __future__ import annotations

from email.mime import text
import math
import re
from statistics import mean


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        return [
            text[i:i + self.chunk_size].strip()
            for i in range(0, len(text), step)
        ]


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3, overlap: int = 0) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)
        self.overlap = max(0, min(overlap, max_sentences_per_chunk - 1))

    def chunk(self, text: str) -> list[str]:
        # TODO: split into sentences, group into chunks
        if not text:
            return []

        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        step = self.max_sentences_per_chunk - self.overlap

        for i in range(0, len(sentences), step):
            group = sentences[i:i + self.max_sentences_per_chunk]
            chunk = " ".join(group)
            if chunk:
                chunks.append(chunk)

        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        # TODO: implement recursive splitting strategy
        if not text:
            return []

        return self._split(text.strip(), self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # TODO: recursive helper used by RecursiveChunker.chunk
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            return [
                current_text[i:i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        sep = remaining_separators[0]

        parts = current_text.split(sep) if sep else list(current_text)

        chunks = []
        buffer = ""

        for part in parts:
            candidate = (buffer + sep + part) if buffer else part

            if len(candidate) <= self.chunk_size:
                buffer = candidate
            else:
                if buffer:
                    chunks.append(buffer.strip())

                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, remaining_separators[1:]))
                    buffer = ""
                else:
                    buffer = part

        if buffer:
            chunks.append(buffer.strip())

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    # TODO: implement cosine similarity formula

    # 1. Check cùng chiều dài
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must be same length")

    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # TODO: call each chunker, compute stats, return comparison dict
        # from chunking import SentenceChunker, RecursiveChunker, FixedSizeChunker

        strategies = {
            "by_sentences": SentenceChunker(),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size),
        }

        return {
            name: self._compute_stats(chunker.chunk(text))
            for name, chunker in strategies.items()
        }

    def _compute_stats(self, chunks: list[str]) -> dict:
        if not chunks:
            return {
                "count": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "chunks": []  
            }

        lengths = [len(c) for c in chunks if c]

        return {
            "count": len(chunks),
            "avg_length": round(mean(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "chunks": chunks 
        }
    
    
