from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        if not store:
            raise ValueError("store must not be None")
        if not callable(llm_fn):
            raise ValueError("llm_fn must be callable")

        self.store = store
        self.llm_fn = llm_fn
        pass

    def answer(self, question: str, top_k: int = 5) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # 1. Retrieve
        results = self.store.search(question, top_k=top_k)

        if not results:
            return "I don't know."

        filtered = [r for r in results if r.get("score", 0) > 0]

        if not filtered:
            return "I don't know."

        MAX_CONTEXT_CHARS = 2000

        context_chunks = []
        total_length = 0

        for i, r in enumerate(filtered):
            chunk = r["content"].strip()

            if not chunk:
                continue

            chunk_text = f"[Source {i+1}]\n{chunk}"

            if total_length + len(chunk_text) > MAX_CONTEXT_CHARS:
                break

            context_chunks.append(chunk_text)
            total_length += len(chunk_text)

        context = "\n\n".join(context_chunks)

        # 2. Build prompt (STRONG anti-hallucination)
        prompt = f"""
    You are a QA assistant using Retrieval-Augmented Generation (RAG).

    STRICT RULES:
    - Answer ONLY using the provided context.
    - Do NOT use prior knowledge.
    - If the answer is not explicitly stated, say: "I don't know."
    - Be concise and factual.
    - Cite sources like [Source 1] when relevant.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

        # 3. Call LLM
        answer = self.llm_fn(prompt).strip()

        # 👉 Safety fallback (anti hallucination guard)
        if not answer:
            return "I don't know."

        return answer
        raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
