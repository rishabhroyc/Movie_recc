"""Structured output of the request parser. Validation here is what keeps bad LLM output harmless."""
from typing import Optional
from pydantic import BaseModel, field_validator, model_validator
from src.config import GENRES

_CANON = {g.lower(): g for g in GENRES}


class UserRequest(BaseModel):
    include_genres: list[str] = []
    exclude_genres: list[str] = []
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    similar_to_title: Optional[str] = None
    off_topic: bool = False
    parse_failed: bool = False          # set by our code when the LLM output was unusable

    @field_validator("include_genres", "exclude_genres")
    @classmethod
    def genres_must_be_real(cls, v):
        out = []
        for g in v:
            key = str(g).strip().lower().replace("children's", "children")
            if key not in _CANON:
                raise ValueError(f"unknown genre {g!r}, allowed: {GENRES}")
            out.append(_CANON[key])
        return out

    @field_validator("year_min", "year_max")
    @classmethod
    def sane_year(cls, v):
        if v is not None and not (1880 <= v <= 2030):
            raise ValueError("year out of range")
        return v

    @field_validator("similar_to_title")
    @classmethod
    def short_title(cls, v):
        return v.strip()[:120] if v else None

    @model_validator(mode="after")
    def years_in_order(self):
        if self.year_min and self.year_max and self.year_min > self.year_max:
            raise ValueError("year_min is after year_max")
        return self
