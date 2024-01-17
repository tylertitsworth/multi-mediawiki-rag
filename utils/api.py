from typing import Optional

from pydantic import BaseModel
from embed import load_config

config = load_config()


class Query(BaseModel):
    """Construct a query for an incoming API request.

    Args:
        BaseModel (BaseModel): FastAPI model object
    """

    prompt: str = config["question"]
    num_sources: Optional[int] = config["settings"]["num_sources"]
    temperature: Optional[float] = config["settings"]["temperature"]
    repeat_penalty: Optional[float] = config["settings"]["repeat_penalty"]
    top_k: Optional[int] = config["settings"]["top_k"]
    top_p: Optional[float] = config["settings"]["top_p"]
