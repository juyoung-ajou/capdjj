# backend/rag_core.py

import os
# 환경 변수 로딩을 위한 라이브러리
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

# .env 파일에서 API 키를 불러옵니다.
load_dotenv()

class RAGService:
    def __init__(self):
        print(" [시스템] OpenAI 모델 및 임베딩 로딩 중...")
        
        # 1. 임베딩 모델 (검색용) - 무료 모델 유지
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        
        # 2. LLM 모델 (답변용) - OpenAI gpt-4o-mini
        # temperature=0: 사실 기반 답변을 위해 창의성 끄기
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0
        )
        
        self.persist_directory = "chroma_db"
        self.collection_name = "rag_collection"
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = None

        try:
            self.client.get_collection(name=self.collection_name)
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            print(f" [시스템] 기존 벡터 DB ('{self.collection_name}')를 연결했습니다.")
        except Exception:
            print(f" [오류] 벡터 DB를 찾을 수 없습니다. python build_db.py를 실행하세요.")

    def get_answer(self, query: str):
        if self.vector_store is None:
            return {"answer": "DB가 없습니다.", "sources": [], "context": ""}
        
        # [변경 1] 문서 검색 개수를 5개 -> 7개로 늘려 비교군 확보
        retrieved_docs = self.vector_store.similarity_search(query, k=7)
        
        # [변경 2] AI가 연도를 구분할 수 있도록 [[출처: 파일명]]을 내용 앞에 붙여줌
        context_list = []
        sources = set()
        
        for doc in retrieved_docs:
            filename = os.path.basename(doc.metadata.get("source", "알 수 없음"))
            sources.add(filename)
            # 예: "[[출처: 2024_요람.pdf]] 문서 내용..."
            context_list.append(f"[[출처: {filename}]]\n{doc.page_content}")

        context = "\n\n".join(context_list)
        sorted_sources = sorted(list(sources))

        # [변경 3] 프롬프트에 '최신 연도 우선' 규칙 추가
        prompt = f"""
        당신은 아주대학교 교과과정 전문 AI 조교입니다. 
        아래 [Context]에 있는 내용만을 근거로 사용하여 사용자의 질문에 친절하게 한국어로 답변하세요.
        
        [Context]
        {context}
        
        [Question]
        {query}
        
        [중요 지침]
        1. Context에 서로 다른 연도(예: 2021년, 2024년)의 자료가 있다면, 반드시 **가장 최신 연도의 파일** 내용을 기준으로 답변하세요.
        2. 과거 자료와 내용이 달라졌다면, "2024년 기준으로는 ~입니다. (2021년에는 ~였습니다)"라고 비교해주면 좋습니다.
        3. Context에 없는 내용은 절대 지어내지 말고, 모르면 "제공된 문서에 해당 내용이 없습니다"라고 답하세요.
        4. 학점, 과목명 등 수치는 정확하게 인용하세요.
        5. 답변 끝에 '참고 자료: [파일명]'을 명시하세요.
        """
        
        # 답변 생성
        response = self.llm.invoke(prompt)
        
        # [변경 4] 토큰 사용량 및 비용 계산 (터미널 출력용)
        usage = response.response_metadata.get('token_usage', {})
        total_tokens = usage.get('total_tokens', 0)
        # gpt-4o-mini 기준 대략적 원화 환산 (환율 등 변동 가능, 참고용)
        cost_krw = total_tokens * 0.00025 
        
        print("\n" + "="*40)
        print(f" 💰 [토큰 정산 - gpt-4o-mini]")
        print(f" - 입력(질문+문서): {usage.get('prompt_tokens')} 토큰")
        print(f" - 출력(답변): {usage.get('completion_tokens')} 토큰")
        print(f" - 합계: {total_tokens} 토큰 (약 {cost_krw:.2f}원)")
        print("="*40 + "\n")
        
        # 결과 반환
        return {
            "answer": response.content,
            "sources": sorted_sources,
            "context": context
        }