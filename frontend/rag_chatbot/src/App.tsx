// frontend/src/App.tsx

import { useState } from 'react';
import './App.css';
import type { Message } from './types'; // 'Message'가 타입이므로 'import type' 사용
import { sendChatQuery } from './api/chatApi';

// 컴포넌트 임포트
import ChatWindow from './components/ChatWindow';
import InputArea from './components/InputArea';

function App() {
  // 상태 관리 (파일 및 업로드 관련 상태 제거)
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  // 채팅 전송 핸들러
  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    
    const currentInput = input;
    setInput(""); // 입력창 초기화
    setLoading(true);

    try {
      const res = await sendChatQuery(currentInput);
      const botMessage: Message = { 
        role: "bot", 
        text: res.answer,
        sources: res.sources // 이제 파일명이 담김
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { role: "bot", text: "에러가 발생했습니다." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* 헤더 */}
      <header className="header">
        <h1>🏫 아주대학교 RAG 챗봇</h1>
        <p>교과과정 질의응답 시스템</p>
      </header>
      
      {/* 설정 섹션 제거 */}

      {/* 1. 채팅창 컴포넌트 */}
      <ChatWindow 
        messages={messages} 
        loading={loading} 
      />

      {/* 2. 입력창 컴포넌트 */}
      <InputArea 
        input={input}
        setInput={setInput}
        onSend={handleSend}
        loading={loading}
      />
    </div>
  );
}

export default App;