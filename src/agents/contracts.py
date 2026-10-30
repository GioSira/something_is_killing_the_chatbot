from pydantic import BaseModel, Field
from typing import List, Literal


class LoreAnswer(BaseModel):

    answer: str = Field(description="Answer in 2-5 sentences, at most 250 words, based only on the passages")
    retrieved: List[str] = Field(default_factory=list, description="list of URLs used to answer")
    confidence: Literal["high", "unsure", "low"] = Field(
        description="high: passages answer fully; unsure: partially; low: little or nothing found"
    )
