"""Qdrant-native hybrid search using prefetch + RRF fusion."""

from __future__ import annotations

from qdrant_client import QdrantClient, models


def hybrid_query(
    client: QdrantClient,
    collection_name: str,
    dense_vector: list[float],
    sparse_vector: models.SparseVector,
    dense_limit: int = 50,
    sparse_limit: int = 50,
    rrf_top_k: int = 20,
    section_filter: str | None = None,
    company_filter: str | None = None,
) -> list[models.ScoredPoint]:
    """Run hybrid dense+sparse search with RRF fusion via Qdrant query_points.

    Uses two prefetch branches (dense and sparse) fused with Reciprocal Rank
    Fusion, optionally filtered by section_type and/or company_name.

    Args:
        client: QdrantClient instance.
        collection_name: Name of the Qdrant collection to query.
        dense_vector: Dense embedding of the query text.
        sparse_vector: Sparse BM25 vector of the query text.
        dense_limit: Max candidates from dense branch.
        sparse_limit: Max candidates from sparse branch.
        rrf_top_k: Number of results after RRF fusion.
        section_filter: If set, restrict to this section_type value.
        company_filter: If set, restrict to this company_name value.

    Returns:
        List of ScoredPoint with payload, ordered by RRF score descending.

    Raises:
        ConnectionError: If Qdrant is unreachable.
        qdrant_client.http.exceptions.UnexpectedResponse: On Qdrant error.
    """
    filter_conditions: list[models.FieldCondition] = []
    if section_filter is not None:
        filter_conditions.append(
            models.FieldCondition(
                key="section_type",
                match=models.MatchValue(value=section_filter),
            )
        )
    if company_filter is not None:
        filter_conditions.append(
            models.FieldCondition(
                key="company_name",
                match=models.MatchValue(value=company_filter),
            )
        )

    query_filter = (
        models.Filter(must=filter_conditions) if filter_conditions else None  # type: ignore[arg-type]
    )

    # Apply filter at both prefetch level (required for local/in-memory mode)
    # and top-level query_filter (server-side optimisation).
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=dense_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=sparse_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=rrf_top_k,
        with_payload=True,
    )

    return response.points
