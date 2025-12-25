# backend/experiment_hybrid_extreme.py

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# ==============================================================================
# 1. 🏭 [데이터 공장] Vector를 멘붕에 빠뜨릴 '복제인간' 데이터 100개
# ==============================================================================
documents = []
doc_ids = []

# 상황: 내용은 토시 하나 안 틀리고 똑같은데, '코드 번호'만 다른 100개의 문서
# Vector 입장에서는 이 100개 문장이 전부 똑같은 점수(유사도 0.999...)로 보임
for i in range(1, 101):
    code = f"CODE_{i:03d}" # 예: CODE_001, CODE_002 ...
    # 문장을 길게 만들어서 '코드'의 비중을 줄임 (Vector가 더 헷갈리게)
    text = (f"아주대학교 수학과 전공 필수 과목 안내입니다. "
            f"이 과목의 관리 코드는 {code} 입니다. "
            f"졸업을 위해 반드시 이수해야 하며, 선수 과목 조건을 확인하시기 바랍니다.")
    
    documents.append(text)
    doc_ids.append(code)

# ==============================================================================
# 2. ❓ [실험 질문] Vector는 찍어야 하고, BM25는 보고 맞추는 질문
# ==============================================================================
test_cases = [
    {"query": "관리 코드 CODE_023 과목 정보 알려줘", "target": "CODE_023"},
    {"query": "필수 과목 CODE_055 내용은 뭐야?", "target": "CODE_055"},
    {"query": "CODE_089 이수 조건 확인", "target": "CODE_089"},
    {"query": "졸업 요건 CODE_007 설명해줘", "target": "CODE_007"},
    {"query": "CODE_099 과목 선수 과목이 뭐야?", "target": "CODE_099"}
]

# ==============================================================================
# 3. ⚙️ RRF 알고리즘 (Hybrid 점수 계산기)
# ==============================================================================
def rrf_score(rank, k=60):
    return 1 / (k + rank)

def run_experiment():
    print("⚡ [극한 실험] Vector를 고장내고 Hybrid로 살리기")
    print(f"📄 데이터: {len(documents)}개의 '거의 똑같은' 문서들 (쌍둥이 데이터)")
    print("   -> 모델 로딩 중...")
    
    # 모델 로드
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
    doc_vectors = embeddings.embed_documents(documents)
    
    tokenized_corpus = [doc.split(" ") for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("-" * 80)
    print(f"| {'질문 (Query)':^20} | {'Vector 순위':^12} | {'BM25 순위':^10} | {'Hybrid 순위':^12} |")
    print("-" * 80)
    
    mrr_vec = 0
    mrr_hyb = 0
    
    for test in test_cases:
        query = test["query"]
        target = test["target"]
        gt_idx = doc_ids.index(target)
        
        # 1. Vector Search
        query_vec = embeddings.embed_query(query)
        vec_scores = cosine_similarity([query_vec], doc_vectors)[0]
        # 점수가 다 비슷해서 정렬 순서가 불안정함 (Noise에 민감)
        vec_indices = np.argsort(vec_scores)[::-1]
        vec_rank = np.where(vec_indices == gt_idx)[0][0] + 1
        
        # 2. BM25 Search
        tokenized_query = query.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1]
        bm25_rank = np.where(bm25_indices == gt_idx)[0][0] + 1
        
        # 3. Hybrid (RRF)
        final_scores = {}
        for doc_idx in range(len(documents)):
            v_r = np.where(vec_indices == doc_idx)[0][0] + 1
            b_r = np.where(bm25_indices == doc_idx)[0][0] + 1
            final_scores[doc_idx] = rrf_score(v_r) + rrf_score(b_r)
            
        sorted_hybrid = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        hybrid_indices = [x[0] for x in sorted_hybrid]
        hybrid_rank = hybrid_indices.index(gt_idx) + 1
        
        # MRR 누적
        mrr_vec += 1 / vec_rank
        mrr_hyb += 1 / hybrid_rank
        
        # 출력
        q_short = query[:18] + ".."
        change = "🔺상승" if hybrid_rank < vec_rank else ("-" if hybrid_rank == vec_rank else "🔻하락")
        print(f"| {q_short:<22} | {vec_rank}위        | {bm25_rank}위       | {hybrid_rank}위 ({change})   |")

    # 최종 결과 계산
    avg_mrr_vec = mrr_vec / len(test_cases)
    avg_mrr_hyb = mrr_hyb / len(test_cases)
    improvement = ((avg_mrr_hyb - avg_mrr_vec) / avg_mrr_vec) * 100 if avg_mrr_vec > 0 else 0
    
    print("-" * 80)
    print(f"📊 최종 성적표")
    print(f"   [Before] Vector Only : {avg_mrr_vec:.4f} (거의 랜덤 찍기)")
    print(f"   [After]  Hybrid (RRF): {avg_mrr_hyb:.4f} (정확히 찾아냄)")
    print(f"   🚀 성능 향상률: {improvement:.1f}% 증가")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()