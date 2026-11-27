
import json
import logging
import re
from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None


def llm():
    global _llm

    if _llm is None:
        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing from your .env file"
            )

        _llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=settings.google_api_key,
        temperature=0,
        max_retries=2,
        timeout=120
)

    return _llm


def invoke_llm(prompt: str) -> str:
    response = llm().invoke([HumanMessage(content=prompt)])

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()

    return str(content).strip()


def parse_json_response(response: str, schema):
    text = response.strip()

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            raise ValueError(
                f"Model did not return valid JSON: {text}"
            )

        data = json.loads(match.group(0))

    return schema.model_validate(data)


def web_search_tool():
    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def route_question(state: AgentState):
    response = invoke_llm(f"""
You are an intelligent router for an enterprise IT support assistant.

Classify the user question into exactly one category.

Use "kb" for:
- Company IT policies
- VPN
- Password reset
- MFA
- Laptop setup
- Software access
- Security
- Email
- Devices
- Technical troubleshooting

Use "direct" only for greetings, thanks, and casual conversation.

Return ONLY valid JSON:
{{"route": "kb"}}

Question: {state['question']}
""")

    decision = parse_json_response(response, RouteDecision)

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}"
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:

    return (
        "retrieve_kb"
        if state["source_used"] == "kb"
        else "direct_answer"
    )


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks"
        ),
    }


def grade_kb(state: AgentState):
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in state["kb_docs"]
    )

    response = invoke_llm(f"""
You are an evidence evaluator for an enterprise IT support assistant.

Evaluate whether the private knowledge base contains enough relevant
information to answer the user's question confidently.

Return "good" only when the evidence directly supports the answer.
Otherwise return "weak".

Return ONLY valid JSON:
{{"grade": "good"}}

Question:
{state['question']}

Private company knowledge base:
{context}
""")

    grade = parse_json_response(response, EvidenceGrade)

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}"
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:

    return (
        "generate_from_kb"
        if state["kb_grade"] == "good"
        else "search_web"
    )


def search_web(state: AgentState):
    result = web_search_tool().invoke(
        {"query": state["current_query"]}
    )

    lines = []
    citations = []

    if isinstance(result, dict):
        if result.get("answer"):
            lines.append(
                "Search answer: " + result["answer"]
            )

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\nURL: {url}\nContent: {content}"
            )

            citations.append({
                "title": title or url,
                "url": url,
                "type": "web",
            })

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search"
        ),
    }


def grade_web(state: AgentState):
    response = invoke_llm(f"""
You evaluate external web search evidence.

Determine whether the evidence is directly relevant and sufficient
to answer the user's question.

Return "good" if sufficient.
Otherwise return "weak".

Return ONLY valid JSON:
{{"grade": "good"}}

Question:
{state['question']}

Web evidence:
{state['web_results']}
""")

    grade = parse_json_response(response, EvidenceGrade)

    return {
        "web_grade": grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {grade.grade.upper()}"
        ),
    }


def after_web(
    state: AgentState,
) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"


def rewrite_query(state: AgentState):
    rewritten = invoke_llm(f"""
Rewrite the following IT support question to improve private
knowledge retrieval and external vendor web search.

Preserve the original intent.
Add useful technical keywords.
Do not answer the question.
Return only the rewritten query.

Question:
{state['question']}
""")

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}"
        ),
    }


def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = invoke_llm(f"""
You are an enterprise IT support copilot.

Answer ONLY using the private company knowledge base below.

Instructions:
- Be concise and actionable.
- Present numbered steps when appropriate.
- Never invent company policies or procedures.
- If information is missing, say so.
- Mention that the answer is based on the company's private knowledge base.

Question:
{state['question']}

Private company knowledge base:
{context}
""")

    citations = []
    seen = set()

    for doc in state["kb_docs"]:
        source = doc.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)

            citations.append({
                "title": source.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB"
        ),
    }


def generate_from_web(state: AgentState):
    answer = invoke_llm(f"""
You are an enterprise IT support copilot.

The private company knowledge base did not contain sufficient evidence.

Answer ONLY using the external web evidence below.

Instructions:
- Clearly identify the information as external web information.
- Warn users to validate instructions with their IT team before
  changing company-managed systems.
- Do not invent facts.

Question:
{state['question']}

External web evidence:
{state['web_results']}
""")

    return {
        "answer": answer,
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH"
        ),
    }


def direct_answer(state: AgentState):
    answer = invoke_llm(f"""
Respond briefly and naturally to this message.
Do not perform knowledge retrieval.

Message:
{state['question']}
""")

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval"
        ),
    }


def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the company "
            "knowledge base or external search to answer confidently. "
            "Please contact the IT help desk or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence"
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, function in nodes.items():
        graph.add_node(name, function)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()


agent_graph = build_graph()


def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)
