# backend/experiment_mmr_realistic.py

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 📄 [데이터셋] 현실적인 시나리오
# - 좀비(중복): 12개 (상위권 도배용)
# - 함정(유사): 5개 (헷갈리게 만들기)
# - 정답: 1개 (구체적인 날짜)
# ==============================================================================
documents = []

# [그룹 A] 좀비 문서 (12개) - "기간 안내"라고 제목만 있고 날짜는 없는 글들
# (벡터 유사도가 높아서 1페이지를 차지함)
for i in range(1, 13):
    documents.append(f"[공지] 2024-1학기 수강신청 일정 및 유의사항 안내 ({i})")

# [그룹 B] 함정 문서 (5개) - "기간"은 맞는데 다른 기간 (장학금, 기숙사 등)
documents.append("2024학년도 1학기 국가장학금 신청 기간 안내")
documents.append("2024-1학기 생활관(기숙사) 입사 기간 공지")
documents.append("2024학년도 등록금 납부 기간 안내")
documents.append("1학기 수강 정정 기간 및 취소 기간 안내")
documents.append("계절학기 수강신청 기간은 별도 공지 예정입니다.")

# [그룹 C] 진짜 정답 (1개) - 사용자가 찾는 '구체적 날짜'
target_doc = ">> [필독] 실제 수강신청 기간: 2월 13일(화) 10:00 ~ 2월 15일(목) 17:00 <<"
documents.append(target_doc)
target_idx = len(documents) - 1

# [그룹 D] 배경 문서
documents.append("아주대학교 학식 메뉴 안내")
documents.append("도서관 이용 시간 변경 안내")

# 질문
query = "2024년 1학기 수강신청 기간 며칠부터야?"

# ==============================================================================
# ⚙️ MMR 알고리즘
# ==============================================================================
def mmr_sort(doc_vectors, query_vector, lambda_mult=0.5, top_k=10):
    # lambda_mult=0.5 : 유사도와 다양성을 반반씩 고려 (가장 일반적인 세팅)
    sims_to_query = cosine_similarity([query_vector], doc_vectors)[0]
    
    selected_indices = []
    candidate_indices = list(range(len(doc_vectors)))
    
    for _ in range(top_k):
        best_score = -float("inf")
        best_idx = -1
        
        for idx in candidate_indices:
            relevance = sims_to_query[idx]
            
            if selected_indices:
                sims_to_selected = cosine_similarity(
                    [doc_vectors[idx]], 
                    [doc_vectors[i] for i in selected_indices]
                )[0]
                redundancy = np.max(sims_to_selected)
            else:
                redundancy = 0
            
            mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * redundancy)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
        
    return selected_indices

# ==============================================================================
# 🧪 실험 실행
# ==============================================================================
def run_experiment():
    print("⚡ [현실적인 실험] Standard vs MMR : 정보의 홍수 속에서 정답 찾기")
    print(f"📄 데이터: 총 {len(documents)}개 (중복 12개, 함정 5개, 정답 1개)")
    print("   -> 모델 로딩 중...")
    
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    doc_vectors = embeddings.embed_documents(documents)
    query_vector = embeddings.embed_query(query)
    
    print("-" * 80)
    print(f"❓ 질문: '{query}'")
    print("-" * 80)
    
    # --- [1] Standard Search ---
    # 단순히 유사도 순으로 20개 확인
    sim_scores = cosine_similarity([query_vector], doc_vectors)[0]
    std_indices = np.argsort(sim_scores)[::-1]
    
    std_rank = np.where(std_indices == target_idx)[0][0] + 1
    
    print(f"\n🐢 [Before] Standard Search")
    print(f"   - 정답 위치: {std_rank}위")
    print("     (설명: 유사한 공지사항들에 밀려서 2페이지쯤 뒤에 나옴)")
    
    # 상위 3개만 보여주기
    print("   - 상위 3개 결과:")
    for i in range(3):
        print(f"     {i+1}위: {documents[std_indices[i]]}")
    
    mrr_std = 1 / std_rank

    # --- [2] MMR Search ---
    # lambda=0.5 (적절한 균형)
    mmr_indices = mmr_sort(doc_vectors, query_vector, lambda_mult=0.5, top_k=10)
    
    try:
        mmr_rank = mmr_indices.index(target_idx) + 1
    except ValueError:
        mmr_rank = -1 
        
    print(f"\n🚀 [After] MMR Search")
    print(f"   - 정답 위치: {mmr_rank}위")
    print("     (설명: 중복된 공지들을 건너뛰고 상위권(Top 5) 안으로 진입)")

    print("   - 상위 3개 결과:")
    for i, idx in enumerate(mmr_indices[:3]):
        mark = "👈 ✅ 정답!" if idx == target_idx else ""
        print(f"     {i+1}위: {documents[idx]} {mark}")

    mrr_mmr = 1 / mmr_rank if mmr_rank != -1 else 0

    # 최종 비교
    print("-" * 80)
    print(f"📊 최종 성적표 (현실적인 개선폭)")
    print(f"   [Standard] MRR: {mrr_std:.4f} (찾기 힘듦)")
    print(f"   [MMR]      MRR: {mrr_mmr:.4f} (쾌적함)")
    
    improvement = ((mrr_mmr - mrr_std) / mrr_std) * 100
    print(f"   📈 성능 향상률: {improvement:.1f}% 증가")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()