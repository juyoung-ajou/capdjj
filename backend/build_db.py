# backend/build_db.py
import os
import glob
import re
from typing import Optional
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import chromadb

# --- 설정 ---
PDF_SOURCE_DIR = "pdf_documents"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "rag_collection"
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 100
# --- 설정 끝 ---

def pdf_to_markdown(pdf_path):
    """
    표와 텍스트를 모두 추출하여 AI가 문맥을 놓치지 않게 함
    """
    full_text = ""
    
    # 표 추출 설정 (복잡한 표도 잘 잡도록 설정값 튜닝)
    table_settings = {
        "vertical_strategy": "lines", 
        "horizontal_strategy": "lines",
        "snap_tolerance": 4,
        "intersection_x_tolerance": 5,
        "intersection_y_tolerance": 5,
    }

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = ""
                
                # 1. [구조화 데이터] 표 추출 시도
                tables = page.extract_tables(table_settings)
                
                if tables:
                    print(f"  [p.{i+1}] 📄 표 {len(tables)}개 발견 (Markdown 변환)")
                    for table in tables:
                        # None 값을 빈 문자열로 치환
                        cleaned_table = [
                            [str(cell).replace('\n', ' ') if cell is not None else "" for cell in row]
                            for row in table
                        ]
                        if not cleaned_table: continue

                        # 마크다운 표 생성
                        headers = cleaned_table[0]
                        markdown_table = "\n\n| " + " | ".join(headers) + " |\n"
                        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                        for row in cleaned_table[1:]:
                            markdown_table += "| " + " | ".join(row) + " |\n"
                        markdown_table += "\n"
                        page_text += markdown_table

                # 2. [시각적 데이터] 텍스트 레이아웃 추출 (핵심! ⭐)
                # 표가 있든 없든 무조건 원본 위치 그대로 텍스트를 한번 더 저장합니다.
                # 이렇게 하면 병합된 셀 때문에 표가 깨져도, AI가 이 텍스트를 보고 정답을 찾습니다.
                raw_text_layout = page.extract_text(layout=True)
                if raw_text_layout:
                    page_text += f"\n\n[원본 텍스트 레이아웃]\n{raw_text_layout}\n"

                # 페이지 번호와 함께 저장
                full_text += f"\n[[페이지: {i+1}]]\n{page_text}"
                
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

    return full_text

def extract_department(filename: str) -> Optional[str]:
    # Extract first "<Korean>학과" from filename.
    match = re.search(r"([가-힣]+학과)", filename)
    return match.group(1) if match else None

def build_vector_db():
    print("="*50)
    if not os.path.isdir(PDF_SOURCE_DIR):
        print(f"'{PDF_SOURCE_DIR}' 폴더가 없습니다.")
        return

    pdf_files = glob.glob(os.path.join(PDF_SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        print("PDF 파일이 없습니다.")
        return

    print(f"총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    embeddings = HuggingFaceEmbeddings(model_name="./my_finetuned_model")
    
    all_docs = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename} ...")
        
        markdown_text = pdf_to_markdown(pdf_path)
        
        if markdown_text:
            department = extract_department(filename)
            metadata = {"source": filename}
            if department:
                metadata["department"] = department
            raw_doc = Document(page_content=markdown_text, metadata=metadata)
            docs = text_splitter.split_documents([raw_doc])
            all_docs.extend(docs)
            print(f" -> {len(docs)}개 청크 생성 완료")

    if not all_docs:
        print("저장할 데이터가 없습니다.")
        return

    print("\n" + "="*50)
    print(f"총 {len(all_docs)}개의 청크를 DB에 저장합니다...")

    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except:
        pass

    Chroma.from_documents(
        documents=all_docs, 
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=client
    )
    
    print("✅ 벡터 DB 생성 완료! (표 구조와 시각적 배치를 모두 학습했습니다)")
    print("="*50)

if __name__ == "__main__":
    build_vector_db()
