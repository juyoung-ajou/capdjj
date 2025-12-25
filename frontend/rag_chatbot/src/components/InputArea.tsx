import { useState, KeyboardEvent } from "react";

// --- [전송 아이콘 SVG] ---
const SendIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z" fill="white"/>
  </svg>
);

interface InputAreaProps {
  onSend: (text: string) => void;
  loading: boolean;
}

export default function InputArea({ onSend, loading }: InputAreaProps) {
  const [text, setText] = useState("");

  const handleSendClick = () => {
    if (text.trim() && !loading) {
      onSend(text);
      setText("");
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !loading) {
      handleSendClick();
    }
  };

  return (
    <section className="input-area-container">
      <div className="input-form">
        <input
          type="text"
          className="chat-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="메시지를 입력하세요..."
          disabled={loading}
        />
        <button 
          className="send-button" 
          onClick={handleSendClick} 
          disabled={loading || !text.trim()}
          aria-label="전송"
        >
          <SendIcon />
        </button>
      </div>
    </section>
  );
}