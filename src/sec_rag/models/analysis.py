from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    company_name: str
    section_type: str
    excerpt: str = Field(description="Verbatim excerpt from source supporting the claim")


class Obligation(BaseModel):
    obligation_type: str = Field(
        description="Category: compensation, non-compete, termination, etc."
    )
    party: str = Field(description="Who bears the obligation: employer or employee")
    description: str = Field(description="Plain-language description of the obligation")
    conditions: str | None = Field(default=None, description="Conditions or triggers, if any")
    citations: list[Citation]


class AnalysisResult(BaseModel):
    query: str
    obligations: list[Obligation]
    summary: str = Field(description="2-3 sentence executive summary")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-assessed confidence")
    source_count: int = Field(description="Number of unique source chunks used")


class RelevanceGrade(BaseModel):
    is_relevant: bool
    reasoning: str
    score: float = Field(ge=0.0, le=1.0)


class InputValidation(BaseModel):
    is_valid: bool
    reason: str = ""
