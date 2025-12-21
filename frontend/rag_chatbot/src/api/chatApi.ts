// frontend/src/api/chatApi.ts
import type { ChatResponse } from '../types';
import axios from 'axios';

// 백엔드 주소 (필요시 변경)
const BASE_URL = "http://localhost:8000";

export const sendChatQuery = async (query: string): Promise<ChatResponse> => {
  const response = await axios.post(`${BASE_URL}/chat`, { query });
  return response.data;
};