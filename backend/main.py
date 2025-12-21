# backend/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import ChatRequest       # 분리한 모델 임포트
from rag_core import RAGService      # 분리한 로직 임포트0   dddddddddddddd

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://0.0.0.0:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG 서비스 시작 (서버 켜질 때 모델 로딩됨)
rag_service = RAGService()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 분리한 로직 호출
        result = rag_service.get_answer(request.query)
        
        if result is None:
            # 이 경우는 rag_service.__init__에서 DB 로드 실패 시 발생 가능
            raise HTTPException(status_code=503, detail="Service Unavailable: Database not loaded.")
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))