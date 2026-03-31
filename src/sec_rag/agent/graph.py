"""Build and compile the LangGraph state machine for contract analysis."""

from functools import partial

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from sec_rag.agent.nodes import (
    evaluate_node,
    general_response_node,
    generate_node,
    retrieve_node,
    rewrite_node,
    route_node,
    should_retrieve,
    should_retry_or_generate,
)
from sec_rag.models.state import QueryState
from sec_rag.retrieval.pipeline import RetrievalPipeline


def build_graph(
    llm: BaseChatModel,
    retrieval_pipeline: RetrievalPipeline,
    max_retries: int = 2,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Construct and compile the Adaptive/Corrective RAG graph.

    Topology::

        START --> route
        route --[extraction]-----> retrieve
        route --[general]--------> general_response --> END
        route --[error]----------> END
        retrieve ------------> evaluate
        evaluate --[relevant]-> generate
        evaluate --[not relevant, retries < 2]-> rewrite
        evaluate --[retries >= 2]-> generate  (best effort)
        evaluate --[error]-------> END
        rewrite -----------------> retrieve   (retry loop)
        generate ----------------> END

    Dependencies are injected into node functions via :func:`functools.partial`.
    """
    graph = StateGraph(QueryState)

    # Register nodes with bound dependencies
    graph.add_node("route", partial(route_node, llm=llm))
    graph.add_node("retrieve", partial(retrieve_node, pipeline=retrieval_pipeline))
    graph.add_node("evaluate", partial(evaluate_node, llm=llm))
    graph.add_node("generate", partial(generate_node, llm=llm))
    graph.add_node("rewrite", partial(rewrite_node, llm=llm))
    graph.add_node("general_response", general_response_node)

    # Edges
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", should_retrieve, {
        "retrieve": "retrieve",
        "general_response": "general_response",
        "end": END,
    })
    graph.add_edge("general_response", END)
    graph.add_edge("retrieve", "evaluate")
    retry_fn = partial(should_retry_or_generate, max_retries=max_retries)
    graph.add_conditional_edges("evaluate", retry_fn, {
        "generate": "generate",
        "rewrite": "rewrite",
        "end": END,
    })
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()
