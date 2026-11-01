from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class LoreAnswer(BaseModel):

    answer: str = Field(description="Answer in 2-5 sentences, at most 250 words, based only on the passages")
    images: Optional[List[str]] = Field(default_factory=list, description="list of images URLs that can be used to represent the answer if present.")
    retrieved: List[str] = Field(default_factory=list, description="list of URLs used to answer")
    confidence: Literal["high", "unsure", "low"] = Field(
        description="high: passages answer fully; unsure: partially; low: little or nothing found"
    )


class Source(BaseModel):
    id: int = Field(description="Source identifier as number")
    url: str = Field(description="Source URL")    

class AnnotatedAnswer(BaseModel):
    answer: str = Field(description="Answer in 2-5 sentences, at most 250 words, based only on the passages")
    images: Optional[List[str]] = Field(default_factory=list, description="List of URLs of the image to include in the answer if present")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer")
    next_question: List[str] = Field(default_factory=list, description="List of follow up questions the user can ask")
