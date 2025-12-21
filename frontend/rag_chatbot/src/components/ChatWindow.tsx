import type { Message } from '../types';

interface ChatWindowProps {
  messages: Message[];
  loading: boolean;
}

export default function ChatWindow({ messages, loading }: ChatWindowProps) {
  return (
    <section className="chat-window">
      {messages.length === 0 && (
        <div style={{ textAlign: 'center', marginTop: '150px', color: '#aaa' }}>
          궁금한 내용을 질문해 보세요!<br/>(예: "졸업 이수 학점은 몇 점이야?")
        </div>
      )}
      
      {messages.map((msg, idx) => (
        <div key={idx} className={`msg-wrapper ${msg.role}`}>
          <div className="message-bubble">
            {msg.text}
          </div>
          
          {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
            <div className="source-container">
              <span>📚 근거 자료:</span>
              {msg.sources.map((page, i) => (
                <span key={i} className="source-badge">
                  p.{page}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
      {loading && <div className="loading-text">답변을 생성하고 있습니다... 🤖</div>}
    </section>
  );
}