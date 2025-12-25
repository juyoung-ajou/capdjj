# backend/experiment_finetuning_effect.py

import numpy as np
import os
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# 1. 🎯 [데이터셋] 학습에 사용했던 PDF 내용과 유사한 문장들
# ==============================================================================
documents = [
    # 0. [정답] 졸업 이수 학점 (학습한 내용)
    "수학과를 졸업하기 위해서는 전공필수와 전공선택을 합쳐 총 120학점을 이수해야 한다.",
    
    # 1. [함정] 일반적인 졸업 얘기 (유사도 높음)
    "대학교 졸업은 인생의 새로운 시작을 의미하며 학위수여식은 매년 2월에 열린다.",
    
    # 2. [정답] 학과 사무실 위치 (학습한 내용)
    "수학과 학과 사무실(행정실)은 팔달관 337호에 위치하고 있다.",
    
    # 3. [함정] 비슷한 위치 (오답)
    "팔달관 338호에는 물리학과 학생회실이 위치하고 있다.",
    
    # 4. [정답] 과목 정보
    "해석개론1 과목의 학수번호는 MATH321이며 3학년 전공필수 과목이다.",
    
    # 5. [기타] 
    "아주대학교는 경기도 수원시에 위치한 사립 대학교이다."
]

# ==============================================================================
# 2. ❓ [실험 질문] 학습 데이터에 있었을 법한 질문들
# ==============================================================================
test_cases = [
    {"query": "졸업하려면 몇 학점 들어야 돼?", "answer_idx": 0},
    {"query": "과사(사무실) 어디에 있어?", "answer_idx": 2},
    {"query": "해석개론1은 무슨 과목이야?", "answer_idx": 4}
]

def run_experiment(model_path, label):
    print(f"\n⚡ [{label}] 모델 로딩 중... ({model_path})")
    
    # 모델 경로 확인
    if model_path.startswith("./") and not os.path.exists(model_path):
        print(f"❌ 오류: '{model_path}' 폴더를 찾을 수 없습니다. 학습이 제대로 완료되었나요?")
        return

    try:
        # 모델 로드
        embeddings = HuggingFaceEmbeddings(model_name=model_path)
        
        # 임베딩 변환
        doc_vectors = embeddings.embed_documents(documents)
        query_vectors = embeddings.embed_documents([t["query"] for t in test_cases])
        
        # 유사도 계산
        similarities = cosine_similarity(query_vectors, doc_vectors)
        
        print("-" * 65)
        print(f"| {'질문 (Query)':^20} | 순위 | 유사도(Score) | 결과 |")
        print("-" * 65)
        
        mrr_sum = 0
        hits = 0
        score_sum = 0 # 정답의 유사도 점수 평균 (자신감 측정용)
        
        for i, test in enumerate(test_cases):
            scores = similarities[i]
            ranked_indices = np.argsort(scores)[::-1]
            
            gt_idx = test["answer_idx"]
            rank = np.where(ranked_indices == gt_idx)[0][0] + 1
            score = scores[gt_idx]
            
            mrr_sum += 1 / rank
            score_sum += score
            
            is_hit = "✅" if rank == 1 else "❌"
            if rank == 1: hits += 1
            
            # 1위가 오답이면 이유 출력
            note = ""
            if rank > 1:
                wrong_idx = ranked_indices[0]
                note = f" (1위 착각: {documents[wrong_idx][:10]}...)"
                
            q_short = test['query'][:18] + ".."
            print(f"| {q_short:<22} | {rank}위   | {score:.4f}        | {is_hit}{note}")

        avg_mrr = mrr_sum / len(test_cases)
        avg_score = score_sum / len(test_cases)
        
        print("-" * 65)
        print(f"📊 최종 성적표 ({label})")
        print(f"   - MRR (평균 순위): {avg_mrr:.4f}")
        print(f"   - 정답 유사도 평균: {avg_score:.4f} (높을수록 확신을 가짐)")
        print("=" * 65)
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    # 1. 기존 모델 (Before): 이미 훌륭하지만, 우리 학교 데이터는 처음 봄
    run_experiment("jhgan/ko-sroberta-multitask", "Before: 일반 SBERT")
    
    # 2. 내 튜닝 모델 (After): 우리 학교 데이터로 '족집게 과외' 받음
    run_experiment("./my_finetuned_model", "After: 나만의 튜닝 모델")