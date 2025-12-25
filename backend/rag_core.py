# backend/rag_core.py

import os
from dotenv import load_dotenv

# 필수 라이브러리 로드 (에러 방지 처리)
try:
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.retrievers import BM25Retriever
    import chromadb
except ImportError as e:
    print(f"[치명적 오류] 필수 라이브러리 누락: {e}")
    raise e

load_dotenv()

class RAGService:
    def __init__(self):
        print(" [시스템] 하이브리드 검색(BM25 + Vector) 엔진 로딩 중...")
        
        # 1. [Triplet Loss 원리] 임베딩 모델
        self.embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.persist_directory = "chroma_db"
        self.collection_name = "rag_collection"
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = None
        
        # 앙상블을 위한 개별 검색기
        self.bm25_retriever = None
        self.chroma_retriever = None

        try:
            # DB 연결
            self.client.get_collection(name=self.collection_name)
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            
            # [BM25 & TF-IDF 원리] 키워드 검색기 구축
            print(" [시스템] BM25 인덱스 구축 및 MMR 검색기 준비...")
            
            existing_data = self.vector_store.get()
            all_docs = existing_data["documents"]
            metadatas = existing_data["metadatas"]
            
            if not all_docs:
                print(" [경고] DB가 비어있습니다.")
                return

            from langchain_core.documents import Document
            doc_objects = []
            for t, m in zip(all_docs, metadatas):
                if m is None: m = {}
                doc_objects.append(Document(page_content=t, metadata=m))
            
            if doc_objects:
                # 1. BM25 검색기 (키워드 매칭 - TF-IDF 확률 통계)
                self.bm25_retriever = BM25Retriever.from_documents(doc_objects)
                self.bm25_retriever.k = 5
                
                # 2. Vector 검색기 (MMR 원리 - 다양성 확보)
                self.chroma_retriever = self.vector_store.as_retriever(
                    search_type="mmr", 
                    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.6}
                )
                
                print(f" [시스템] 앙상블(Ensemble) 검색 엔진 준비 완료! (BM25 + MMR)")
            
        except Exception as e:
            print(f" [오류] 초기화 실패: {e}")

    def get_answer(self, query: str):
        if self.bm25_retriever is None or self.chroma_retriever is None:
            return {"answer": "시스템이 초기화되지 않았습니다.", "sources": [], "context": ""}
        
        try:
            # [핵심] 앙상블(Ensemble) 로직 - RRF 방식 응용
            # 1. 키워드 검색 (BM25)
            bm25_docs = self.bm25_retriever.invoke(query)
            
            # 2. 의미 검색 (Vector + MMR)
            vector_docs = self.chroma_retriever.invoke(query)
            
            # 3. 결과 섞기 (Vector 1등 -> BM25 1등 -> Vector 2등 -> ...)
            combined_docs = []
            seen_contents = set()
            
            max_len = max(len(bm25_docs), len(vector_docs))
            for i in range(max_len):
                if i < len(vector_docs):
                    doc = vector_docs[i]
                    if doc.page_content not in seen_contents:
                        combined_docs.append(doc)
                        seen_contents.add(doc.page_content)
                
                if i < len(bm25_docs):
                    doc = bm25_docs[i]
                    if doc.page_content not in seen_contents:
                        combined_docs.append(doc)
                        seen_contents.add(doc.page_content)
            
            # 상위 7개 선택
            final_docs = combined_docs[:7]
            
            # 컨텍스트 조립
            context_list = []
            sources = set()
            
            for doc in final_docs:
                filename = os.path.basename(doc.metadata.get("source", "알 수 없음"))
                sources.add(filename)
                context_list.append(f"[[출처: {filename}]]\n{doc.page_content}")

            context = "\n\n".join(context_list)
            sorted_sources = sorted(list(sources))

            # [수정됨] 프롬프트에 가독성 관련 지시사항 추가
            prompt = f"""
            당신은 아주대학교 교과과정 전문 AI 조교입니다. 
            아래 [Context]에 있는 내용만을 근거로 사용하여 사용자의 질문에 친절하게 한국어로 답변하세요.
            
            [Context]
            {context}
            
            [Question]
            {query}
            
            [지침]
            1. 답변은 **가독성** 있게 작성하세요.
            2. 나열되는 정보는 반드시 **불릿 포인트(-)**나 **숫자 리스트**를 사용하여 줄바꿈을 하세요.
            3. 핵심 키워드나 과목명은 **굵게(Bold)** 표시하세요. (예: **해석개론**)
            4. Context에 서로 다른 연도(예: 2021년, 2024년)가 있다면 **최신 연도**를 우선하세요.
            5. 답변 끝에 '참고 자료: [파일명]'을 명시하세요.
            """
            
            response = self.llm.invoke(prompt)
            
            # 토큰 정산 출력
            usage = response.response_metadata.get('token_usage', {})
            total = usage.get('total_tokens', 0)
            cost = total * 0.00025
            print(f"\n 💰 [토큰 정산] 합계: {total} (약 {cost:.2f}원)")

            return {
                "answer": response.content,
                "sources": sorted_sources,
                "context": context
            }
            
        except Exception as e:
            print(f" [오류] 답변 생성 중 문제 발생: {e}")
            return {"answer": "오류가 발생했습니다.", "sources": [], "context": ""}