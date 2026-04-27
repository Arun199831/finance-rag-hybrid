import time
from pydantic import BaseModel, Field
from fastapi import FastAPI,HTTPException

class AskRequest(BaseModel):
    question: str =Field(...,min_length=5) 
    top_k: int =Field(ge=1,le=10,default=4)
    
class AskResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    
app = FastAPI(title="Practice RAG API")
    
    
@app.get("/health")
async def health():
    try:
        return{"status":"ok"}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.post("/ask",response_model=AskResponse)
async def ask(request:AskRequest):
    start_time=time.time()
    try:
        answer = "this is test answer "
        
        latency_ms = (time.time() -start_time) *1000
        
        return AskResponse (
            question = request.question,
            answer = answer,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
        