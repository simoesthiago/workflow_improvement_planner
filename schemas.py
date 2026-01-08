from __future__ import annotations

from pydantic import BaseModel, Field


class PlannerState(BaseModel):
    case_id: str

    assessment: dict = Field(default_factory=dict)
    context_text: str = ""
    context_warnings: list[str] = Field(default_factory=list)

    solution_md: str | None = None
    prototype_md: str | None = None

    external_sources: list[dict] = Field(default_factory=list)

    run_id_solution: str | None = None
    run_id_prototype: str | None = None
    export_ts: str | None = None

    errors: list[str] = Field(default_factory=list)

    def append_error(self, msg: str) -> None:
        self.errors.append(msg)
