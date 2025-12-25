import { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import InputArea from "./components/InputArea";
import { Message } from "./types";
import "./App.css";

// --- [로고 SVG 컴포넌트] ---
const AjouBotLogo = () => (
  <svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" className="logo-icon">
    <circle cx="20" cy="20" r="20" fill="url(#paint0_linear)"/>
    <path d="M20 8L8 28H13L15.5 23H24.5L27 28H32L20 8Z" fill="white"/>
    <path d="M17 20H23" stroke="#D4AF37" strokeWidth="2"/>
    <defs>
      <linearGradient id="paint0_linear" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
        <stop stopColor="#003366"/>
        <stop offset="1" stopColor="#0055A5"/>
      </linearGradient>
    </defs>
  </svg>
);

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (text: string) => {
    const userMsg: Message = { role: "user", text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      const data = await response.json();
      const botMsg: Message = {
        role: "bot",
        text: data.answer,
        sources: data.sources,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMsg: Message = {
        role: "bot",
        text: "죄송합니다. 서버 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요.",
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* 헤더 섹션 */}
      <header className="app-header">
        <div className="branding">
          <AjouBotLogo />
          <div className="brand-text">
            <h1 className="brand-title">아주대학교 교과과정 AI</h1>
            <span className="brand-subtitle">Ajou University Curriculum Chatbot</span>
          </div>
        </div>
        <nav className="header-nav">
          <span className="nav-item active">챗봇 서비스</span>
          <span className="nav-item">이용 안내</span>
          <span className="nav-item">포탈 바로가기</span>
        </nav>
      </header>

      {/* 메인 콘텐츠 카드 */}
      <main className="main-content-card">
        <ChatWindow messages={messages} loading={loading} onSuggestionClick={handleSend} />
        <InputArea onSend={handleSend} loading={loading} />
      </main>
    </div>
  );
}

export default App;