from pydantic import BaseModel,Field

class AskRequest(BaseModel):
    question : str =Field (...,min_length=5)
    top_k : int = Field(default=4,le=10,ge=1)
    
class AskResponse(BaseModel):
    question : str 
    answer : str
    latency_ms : float