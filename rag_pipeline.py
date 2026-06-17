import json
import os
import re

from dotenv import load_dotenv
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from embeddings import get_embeddings

load_dotenv()

# ─── LLM config ───────────────────────────────────────────────
LLM_MODEL = "gemini-2.5-flash"

# ─── Domain-agnostic Q&A prompt ───────────────────────────────
PROMPT_TEMPLATE = """
You are an expert academic research assistant helping scholars understand
research papers from any field (science, engineering, humanities, social
sciences, medicine, and more).

Use ONLY the context below to answer the question.
If the answer is not clearly supported by the context, say:
"I could not find this information in the uploaded documents."

When asked about findings or conclusions, summarize the most important
result described in the context, even if the exact phrase is not used.

Be precise, objective, and cite specific details from the context when available.

Context:
{context}

Question: {question}

Answer:
"""

# ─── Paper brief extraction prompt ────────────────────────────
PAPER_BRIEF_PROMPT = """You are an expert academic reviewer. Analyze the uploaded research 
paper(s) and extract the following in a structured format:

1. RESEARCH PROBLEM: What problem does this paper solve?
2. NOVELTY: What is new or unique about this paper compared to 
   prior work? What does it contribute that didn't exist before?
3. METHODOLOGY: What methods, models, or techniques were used?
4. KEY FINDINGS: What were the main results or discoveries?
5. LIMITATIONS: What are the weaknesses or limitations acknowledged?
6. FUTURE WORK: What do the authors suggest as next steps?
7. REAL WORLD IMPACT: How could this research be applied in practice?

Be specific. Pull exact numbers, model names, and dataset names 
where available. If a section is not discussed in the paper, 
say 'Not explicitly mentioned'."""

BRIEF_KEYS = [
    "research_problem",
    "novelty",
    "methodology",
    "key_findings",
    "limitations",
    "future_work",
    "real_world_impact",
]

NOT_FOUND_PHRASES = [
    "i could not find",
    "not in the documents",
    "not in the uploaded documents",
]


def _get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def load_vectorstore():
    """Load the persisted FAISS index with HuggingFace embeddings."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _extract_sources(source_documents) -> list[dict]:
    """Convert retrieved LangChain documents into citation dicts."""
    sources = []
    for doc in source_documents:
        page = doc.metadata.get("page", "N/A")
        if isinstance(page, int):
            page = page + 1

        source = {
            "file": os.path.basename(doc.metadata.get("source", "Unknown")),
            "page": page,
            "snippet": doc.page_content[:200] + "...",
        }
        if source not in sources:
            sources.append(source)
    return sources


def _is_not_found(answer: str) -> bool:
    """Detect when the RAG pipeline could not answer from context."""
    lower = answer.lower()
    return any(phrase in lower for phrase in NOT_FOUND_PHRASES)


def _rewrite_question(llm: ChatGoogleGenerativeAI, question: str) -> str:
    """Rephrase a vague question so it is easier to retrieve from academic text."""
    rewrite_prompt = (
        "Rephrase this research question to be more specific and "
        f"extractable from academic text: {question}\n\n"
        "Return only the rephrased question, nothing else."
    )
    response = llm.invoke(rewrite_prompt)
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return content.strip() or question


def _run_rag(question: str) -> dict:
    """Core RAG call: retrieve context, generate answer, return sources."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )

    llm = _get_llm()
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": _extract_sources(result["source_documents"]),
    }


def get_answer_with_sources(question: str) -> dict:
    """
    Answer a question with cited sources.
    Self-corrects by rewriting the query once if the first attempt finds nothing.
    """
    try:
        result = _run_rag(question)

        # ─── Self-correction: retry with a rewritten question ───
        if _is_not_found(result["answer"]):
            llm = _get_llm()
            rewritten = _rewrite_question(llm, question)
            retry_result = _run_rag(rewritten)

            if not _is_not_found(retry_result["answer"]):
                retry_result["answer"] += " (Answer found after query rewriting)"
                return retry_result

        return result

    except Exception as e:
        return {
            "answer": f"An error occurred while answering your question: {e}",
            "sources": [],
        }


def _parse_brief_json(text: str) -> dict:
    """Parse LLM JSON output into the 7-section brief dict."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {
                key: parsed.get(key, "Not explicitly mentioned")
                for key in BRIEF_KEYS
            }
    except json.JSONDecodeError:
        pass

    # Fallback: return raw text under research_problem if JSON parsing fails
    return {key: "Not explicitly mentioned" for key in BRIEF_KEYS} | {
        "research_problem": text.strip() or "Could not parse paper brief.",
    }


def generate_paper_brief() -> dict:
    """
    Auto-extract a structured 7-section summary from indexed papers.
    Uses the vectorstore retriever (k=6) to gather context, then Gemini to analyze.
    """
    default = {key: "Not explicitly mentioned" for key in BRIEF_KEYS}

    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )

        # Broad retrieval query to capture problem, methods, results, and limitations
        docs = retriever.invoke(
            "research problem novelty methodology key findings limitations "
            "future work real world impact results conclusions"
        )
        context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        llm = _get_llm()
        full_prompt = (
            f"{PAPER_BRIEF_PROMPT}\n\n"
            f"Context from the uploaded paper(s):\n{context}\n\n"
            "Return ONLY valid JSON (no markdown fences) with exactly these keys:\n"
            "research_problem, novelty, methodology, key_findings, "
            "limitations, future_work, real_world_impact"
        )

        response = llm.invoke(full_prompt)
        content = response.content if hasattr(response, "content") else str(response)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        return _parse_brief_json(content)

    except Exception as e:
        return default | {"research_problem": f"Error generating paper brief: {e}"}


def get_indexed_paper_count() -> int:
    """Count PDF files in the data folder (used for Compare Papers tab visibility)."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return 0
    return len([f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")])
