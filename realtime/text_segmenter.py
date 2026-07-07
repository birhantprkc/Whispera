from __future__ import annotations


class StreamingTextSegmenter:
    """Cut incremental LLM text into TTS-friendly chunks.

    Each chunk is synthesized as an independent ``stream_tts`` call, so cutting
    inside a sentence breaks the prosodic continuity of that sentence (the two
    halves are generated without shared acoustic context). To keep a sentence
    coherent we only break on sentence-final punctuation, so a whole sentence is
    synthesized in one pass.

    A single sentence longer than ``hard_limit`` characters with no
    sentence-final punctuation is force-flushed as a safeguard against never
    emitting audio; the cut backs off to the last whitespace so a word is never
    split (see ``_flush_word_safe``).
    """

    def __init__(self, hard_limit: int = 160):
        self.hard_limit = hard_limit
        self.buffer = ""
        self.sentence_punct = set(".!?。！？")

    def feed(self, text_delta: str) -> list[str]:
        chunks: list[str] = []
        if not text_delta:
            return chunks
        for char in text_delta:
            self.buffer += char
            if char in self.sentence_punct:
                chunk = self._flush()
                if chunk:
                    chunks.append(chunk)
            elif len(self.buffer.strip()) >= self.hard_limit:
                chunk = self._flush_word_safe()
                if chunk:
                    chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        return self._flush()

    def reset(self) -> None:
        self.buffer = ""

    def _flush(self) -> str:
        value = self.buffer.strip()
        self.buffer = ""
        return value

    def _flush_word_safe(self) -> str:
        """Force-flush at ``hard_limit`` without splitting a word.

        Back off to the last whitespace so only whole words are emitted and keep
        the trailing partial word in the buffer for the next chunk. If there is
        no whitespace (e.g. CJK text, or a single very long token) fall back to a
        plain flush.
        """
        stripped = self.buffer.strip()
        cut = stripped.rfind(" ")
        if cut <= 0:
            self.buffer = ""
            return stripped
        head = stripped[:cut].strip()
        self.buffer = stripped[cut + 1:]
        return head
