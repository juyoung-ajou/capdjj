# backend/train_model.py

import json
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader

# 설정
BASE_MODEL = "jhgan/ko-sroberta-multitask" # 기존에 쓰던 모델
TRAIN_DATA_FILE = "triplet_train_data.json"
OUTPUT_PATH = "./my_finetuned_model" # 학습된 모델이 저장될 폴더
EPOCHS = 3 # 학습

def train():
    print("🔥 모델 학습 준비 중...")
    
    # 1. 모델 로드
    word_embedding_model = models.Transformer(BASE_MODEL, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # 2. 데이터 로드
    with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    train_examples = []
    for item in data:
        # Triplet 구조: [질문, 정답(Positive), 오답(Negative)]
        train_examples.append(InputExample(texts=[item['anchor'], item['positive'], item['negative']]))
        
    # 데이터 로더 생성
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # 3. ⭐ 핵심: Triplet Loss 함수 적용 ⭐
    # (Anchor와 Positive는 가깝게, Anchor와 Negative는 멀게 만드는 함수)
    train_loss = losses.TripletLoss(model=model)
    
    print(f"🚀 학습 시작! (데이터 개수: {len(train_examples)}, Epochs: {EPOCHS})")
    print("   (컴퓨터 사양에 따라 시간이 걸릴 수 있습니다...)")
    
    # 4. 학습 실행
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        output_path=OUTPUT_PATH,
        show_progress_bar=True
    )
    
    print(f"✅ 학습 완료! 나만의 모델이 '{OUTPUT_PATH}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    train()