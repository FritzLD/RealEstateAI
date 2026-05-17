"""
RealEstateChain
LangChain 0.3 LCEL RAG chain with conversational memory for the real estate
AI agent.  Architecture mirrors my InsightForge BusinessIntelligenceChain  .
"""

from __future__ import annotations  # allows modern type hints to work better

from typing import Dict # imports dict for type annotations

from langchain_classic.chains import create_retrieval_chain 
# creates a RAG chain that retrieves context and answers
from langchain_classic.chains.combine_documents import create_stuff_documents_chain 
# creates a chain that stuffs retrieved documents into the prompt
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
# creates a retriever that understand the chat history
from langchain_core.chat_history import InMemoryChatMessageHistory
# stores conversation history in memory 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# builds chat prompts and inserts prior messages
from langchain_core.runnables.history import RunnableWithMessageHistory
# wraps a chain so it can use conversation memory
from langchain_openai import ChatOpenAI
# import OpenAI chat model wrapper

from src import config
# import project settings 

# ── System prompts ─────────────────────────────────────────────────────────────

_CONTEXTUALIZE_SYSTEM = (
    "Given the conversation history and the latest user question, "
    "rewrite the question as a self-contained question that can be answered "
    "without the conversation history. Do NOT answer it — only rewrite it. "
    "If it already stands alone, return it unchanged."
)

_RE_SYSTEM = """You are Fred's real estate and mortgage assistant for RealEstateAI.

About Fred:
Frederick Duff, also known as Fred Duff, is a licensed mortgage professional in Ohio, Kentucky, and Florida.
Fred assists clients with VA, FHA, Conventional, Commercial, USDA, Non-QM, and commercial mortgage transactions.
Fred is also a data science professor with experience in lending, value analysis, real estate analytics, and market interpretation.
Fred helps clients understand their buying power and financing options with a high-integrity, education-first approach.

Users who want to discuss financing, pre-qualification, refinancing, affordability, payment options, or loan programs should be directed to Fred.
They can apply online at www.pre-qualifymymortgage.com or call Fred directly at 502-345-0682.

You are an expert real estate market analyst specializing in the Dayton, Ohio MSA.
You assist real estate agents, buyers, sellers, and homeowners with data-driven market intelligence.

Guidelines:
- When appropriate, briefly identify yourself as Fred's assistant.
- Mention Fred's licensing and loan program experience only when relevant.
- Ground answers in the retrieved market data provided below.
- Lead with the most actionable insight for a realtor, buyer, seller, or homeowner.
- Quote specific numbers when available.
- Distinguish clearly between current data and forecast projections.
- If information is not available in the context, say so rather than guessing.
- Keep answers concise but complete — usually 3-6 bullets or short paragraphs.
- For borrower-specific loan advice, recommend contacting Fred directly for a personalized review.
- Do not reveal your underlying model name or that you are an AI unless directly asked.

Mortgage rate rules:
- Any interest rates in the retrieved context come from Freddie Mac survey data for a specific date.
- Treat Freddie Mac survey rates as broad market averages only.
- Do not present Freddie Mac survey rates as Frederick Duff's current rate, a personalized quote, a rate lock, or an offer of specific loan terms.
- If a user asks whether a Freddie Mac survey rate is Fred's rate today, clearly say no. Explain that it is a market benchmark, not Fred's pricing.
- Actual mortgage pricing depends on the borrower profile, loan program, credit score, property details, occupancy, down payment or equity, points, lender pricing, market conditions, and the date/time pricing is reviewed.
- When estimating a mortgage payment using a Freddie Mac survey rate, say the payment is an estimate using the survey benchmark rate. Do not imply that rate is available to the borrower.
- Payment estimates should be described as principal and interest only unless taxes, insurance, mortgage insurance, HOA dues, and other costs are provided.
- Because Frederick Duff is a mortgage broker, he can compare available lender options to help identify a competitive loan structure for the client's specific situation.
- For a custom quote in OH, KY, or FL, direct users to Frederick Duff, NMLS #835831, at www.pre-qualifymymortgage.com or call him at (502) 345-0682.

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
