"""
RealEstateChain
LangChain 0.3 LCEL RAG chain with conversational memory for the real estate
AI agent.  Architecture mirrors InsightForge's BusinessIntelligenceChain exactly.
"""

from __future__ import annotations

from typing import Dict

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from src import config

# ── System prompts ─────────────────────────────────────────────────────────────

_CONTEXTUALIZE_SYSTEM = (
    "Given the conversation history and the latest user question, "
    "rewrite the question as a self-contained question that can be answered "
    "without the conversation history. Do NOT answer it — only rewrite it. "
    "If it already stands alone, return it unchanged."
)

_RE_SYSTEM = """You are an expert real estate market analyst and mortgage consultant
specialising in the Dayton, Ohio MSA (Metropolitan Statistical Area).  You assist
real estate agents and their clients with data-driven market intelligence.

Guidelines:
- Ground every answer in the retrieved market data provided below.
- Lead with the most actionable insight for a realtor or home buyer/seller.
- Quote specific numbers (rates, unit counts, percentages) when available.
- Distinguish clearly between current data and forecast projections.
- When discussing mortgage rates and refinancing, include estimated monthly savings.
- If information is not available in the context, say so rather than guessing.
- Keep answers concise but complete — aim for 3-6 bullet points or short paragraphs.
- Do not reveal your underlying model name or that you are an AI unless directly asked.

Retrieved market context:
{context}
"""


class RealEstateChain:
    """
    Conversational RAG chain for real estate market intelligence.

    Parameters
    ----------
    api_key    : OpenAI API key
    model_name : LLM model name (default gpt-4o-mini)
    retriever  : LangChain retriever (typically a RealEstateRetriever instance)
    temperature: LLM temperature
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = config.DEFAULT_MODEL,
        retriever=None,
        temperature: float = 0.0,
    ) -> None:
        self._llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key,
        )
        self._retriever = retriever
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
        self._chain = self._build_chain()

    # ── Chain construction ────────────────────────────────────────────────────

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        return self._store[session_id]

    def _build_chain(self) -> RunnableWithMessageHistory:
        # 1. Contextualise prompt — reformulates question given chat history
        ctx_prompt = ChatPromptTemplate.from_messages([
            ("system", _CONTEXTUALIZE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self._llm, self._retriever, ctx_prompt
        )

        # 2. QA prompt — answers using retrieved context
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", _RE_SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(self._llm, qa_prompt)

        # 3. Retrieval chain
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        # 4. Wrap with memory
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def ask(self, question: str, session_id: str = "default") -> str:
        """Send a question and return the agent's answer."""
        result = self._chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return result.get("answer", "")

    def clear_history(self, session_id: str = "default") -> None:
        """Clear conversation history for a session."""
        if session_id in self._store:
            self._store[session_id].clear()

    def get_history(self, session_id: str = "default") -> list:
        """Return list of message dicts for a session."""
        hist = self._store.get(session_id)
        if not hist:
            return []
        return [
            {"role": "human" if m.type == "human" else "assistant", "content": m.content}
            for m in hist.messages
        ]
