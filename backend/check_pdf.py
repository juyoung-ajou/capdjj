# check_pdf.py
from langchain_community.document_loaders import PyPDFLoader

# 확인하고 싶은 PDF 파일 경로를 넣으세요
pdf_path = "수학과 (1).pdf" 

try:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print("\n" + "="*50)
    print(f"총 {len(pages)} 페이지를 읽었습니다.")
    print("="*50)
    
    # 첫 페이지와 중간 페이지 내용 출력
    print("[1페이지 내용 미리보기]")
    print(pages[0].page_content[:300]) # 앞부분 300자만 출력
    print("\n" + "-"*50 + "\n")
    
    if len(pages) > 2:
        print("[3페이지 내용 미리보기]")
        print(pages[2].page_content[:300])
        print("="*50)

except Exception as e:
    print(f"에러 발생: {e}")