// frontend/src/types/index.ts

// 1. 메시지 타입 (export 필수!)
export interface Message {
  role: string;
  text: string;
  sources?: number[];
}

// 2. 채팅 응답 타입 (여기가 문제였음, export 필수!)
export interface ChatResponse {
  answer: string;
  sources: number[];
  context: string;
}

// 3. 업로드 응답 타입 (export 필수!)
export interface UploadResponse {
  message: string;
  chunks: number;
  filename: string;
}