"""
api/models.py
==============
Pydantic request / response models for the FastAPI backend.
Kept separate from api.py so each file stays under 80 lines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    description: str = Field(
        ...,
        min_length=5,
        example="White male, 40s, square jaw, brown hair, scar on left cheek",
    )
    use_llm: bool = Field(True, description="Use Groq LLM (True) or rule-based (False)")


class ParseResponse(BaseModel):
    attributes: dict
    parser_used: str
    non_null_count: int   # how many fields were actually detected


class GenerateRequest(BaseModel):
    description: str = Field(..., min_length=5)
    style: str = Field(
        "forensic_sketch",
        description="forensic_sketch | photorealistic | composite",
    )
    num_images:     int   = Field(2,    ge=1,  le=4)
    seed:           Optional[int] = Field(None, description="Fixed seed for reproducibility")
    use_llm:        bool  = Field(True)
    validate_faces: bool  = Field(True, description="Reject images with no detected face")


class GenerateResponse(BaseModel):
    images:                  List[str]   # base64 PNG
    images_generated:        int
    attributes:              dict
    prompt:                  str
    negative_prompt:         str
    generation_time_seconds: float
    backends_tried:          List[str]   # which backends were attempted
