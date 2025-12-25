# backend/make_triplets.py

import json
import glob
import random
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PDF 폴더 경로
PDF_FOLDER = "pdf_documents"
OUTPUT_FILE = "triplet_train_data.json"

def extract_text_from_pdfs():
    texts = []
    files = glob.glob(f"{PDF_FOLDER}/*.pdf")
    print(f"📂 PDF 파일 {len(files)}개를 읽어옵니다...")
    
    for file in files:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text) > 50: # 너무 짧은 건 버림
                    texts.append(text)
    return texts

def generate_qa_pairs(text_chunk):
    # GPT에게 시킵니다: "이 텍스트를 보고 질문과 정답을 만들어줘"
    prompt = f"""
    아래 텍스트를 읽고, 내용을 잘 반영하는 '질문(Question)'과 그에 대한 '답변(Answer)'을 1개만 만들어줘.
    형식은 JSON으로: {{"Q": "질문내용", "A": "답변내용"}}
    
    [텍스트]
    {text_chunk[:1000]}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 학습 데이터 생성은 mini로 해도 충분
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

def main():
    chunks = extract_text_from_pdfs()
    triplets = []
    
    print(f"🚀 학습 데이터 생성 시작 (총 {len(chunks)}개 문단)...")
    
    for i, chunk in enumerate(chunks):
        # 1. Anchor(질문) & Positive(정답) 생성
        qa = generate_qa_pairs(chunk)
        if not qa: continue
        
        anchor = qa['Q']
        positive = chunk # 혹은 qa['A']를 써도 되지만, 검색엔진은 '원문'을 찾는게 목표이므로 chunk가 좋음
        
        # 2. Negative(오답) 생성 (랜덤하게 다른 문단을 가져옴)
        # (현재 문단이 아닌 다른 문단을 오답으로 간주)
        negative = random.choice(chunks)
        while negative == chunk: # 혹시라도 똑같은 거 뽑으면 다시 뽑기
            negative = random.choice(chunks)
            
        triplets.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative
        })
        
        if i % 10 == 0: print(f"   -> {i}번째 데이터 생성 중...")

    # 저장
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(triplets, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 데이터 생성 완료! '{OUTPUT_FILE}'에 저장되었습니다.")

if __name__ == "__main__":
    main()