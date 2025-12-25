import { useEffect, useRef } from "react";
import type { Message } from "../types";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// --- [SVG 아이콘 컴포넌트] ---
const BotIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 4C16.42 4 20 7.58 20 12C20 16.42 16.42 20 12 20C7.58 20 4 16.42 4 12C4 7.58 7.58 4 12 4Z" fill="#004EA2"/>
    <path d="M8 11C8.55228 11 9 10.5523 9 10C9 9.44772 8.55228 9 8 9C7.44772 9 7 9.44772 7 10C7 10.5523 7.44772 11 8 11Z" fill="#004EA2"/>
    <path d="M16 11C16.5523 11 17 10.5523 17 10C17 9.44772 16.5523 9 16 9C15.4477 9 15 9.44772 15 10C15 10.5523 15.4477 11 16 11Z" fill="#004EA2"/>
    <path d="M12 14C10.33 14 8.85 14.83 8 16H16C15.15 14.83 13.67 14 12 14Z" fill="#004EA2"/>
  </svg>
);

const UserIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 12C14.21 12 16 10.21 16 8C16 5.79 14.21 4 12 4C9.79 4 8 5.79 8 8C8 10.21 9.79 12 12 12ZM12 14C9.33 14 4 15.34 4 18V20H20V18C20 15.34 14.67 14 12 14Z" fill="#D4AF37"/>
  </svg>
);

const WelcomeLogo = () => (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg" className="welcome-logo">
    <circle cx="40" cy="40" r="40" fill="url(#paint0_linear_welcome)" fillOpacity="0.1"/>
    <path d="M40 16L16 56H26L31 46H49L54 56H64L40 16Z" fill="#003366"/>
    <path d="M34 40H46" stroke="#D4AF37" strokeWidth="4"/>
    <defs>
    <linearGradient id="paint0_linear_welcome" x1="0" y1="0" x2="80" y2="80" gradientUnits="userSpaceOnUse">
    <stop stopColor="#003366"/>
    <stop offset="1" stopColor="#0055A5"/>
    </linearGradient>
    </defs>
    </svg>
);


interface ChatWindowProps {
  messages: Message[];
  loading: boolean;
  onSuggestionClick: (text: string) => void; // 추천 질문 클릭 핸들러 추가
}

export default function ChatWindow({ messages, loading, onSuggestionClick }: ChatWindowProps) {
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const suggestions = [
    "수학과 졸업 이수 학점 알려줘",
    "해석개론 선수과목이 뭐야?",
    "장학금 신청 기간은 언제야?",
  ];

  return (
    <section className="chat-window">
      {messages.length === 0 && (
        <div className="welcome-screen">
          <WelcomeLogo />
          <h2 className="welcome-title">무엇을 도와드릴까요?</h2>
          <p className="welcome-desc">
            아주대학교 교과과정에 대해 궁금한 점을<br />자유롭게 질문해주세요.
          </p>
          <div className="example-queries">
            {suggestions.map((text, idx) => (
              <button key={idx} className="query-chip" onClick={() => onSuggestionClick(text)}>
                {text}
              </button>
            ))}
          </div>
        </div>
      )}

      {messages.map((msg, idx) => (
        <div key={idx} className={`msg-wrapper ${msg.role}`}>
          {/* 아바타 추가 */}
          <div className={`avatar ${msg.role}`}>
            {msg.role === "bot" ? <BotIcon /> : <UserIcon />}
          </div>
          
          <div className="message-bubble">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.text}
            </ReactMarkdown>
            
            {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
              <div className="source-container">
                <span>📚 참고 자료:</span>
                {msg.sources.map((page, i) => (
                  <span key={i} className="source-badge">
                    {page.replace(".pdf", "")}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}

      {/* 로딩 인디케이터 (타이핑 애니메이션) */}
      {loading && (
        <div className="loading-indicator">
            <div className={`avatar bot`}>
                <BotIcon />
            </div>
            <div className="typing-bubble">
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
                <div className="typing-dot"></div>
            </div>
        </div>
      )}
      <div ref={chatEndRef} />
    </section>
  );
}