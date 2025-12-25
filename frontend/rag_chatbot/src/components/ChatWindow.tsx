// frontend/src/components/ChatWindow.tsx

import type { Message } from '../types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

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
            {/* 텍스트를 그냥 출력하지 않고 ReactMarkdown으로 감싸서 렌더링 */}
            <ReactMarkdown 
              remarkPlugins={[remarkGfm]}
              components={{
                // 리스트 스타일이 안 먹힐 경우를 대비해 기본 스타일 지정
                ul: ({node, ...props}) => <ul style={{ paddingLeft: '20px', margin: '5px 0' }} {...props} />,
                ol: ({node, ...props}) => <ol style={{ paddingLeft: '20px', margin: '5px 0' }} {...props} />,
                p: ({node, ...props}) => <p style={{ margin: '5px 0' }} {...props} />
              }}
            >
              {msg.text}
            </ReactMarkdown>
          </div>
          
          {msg.role === "bot" && msg.sources && msg.sources.length > 0 && (
            <div className="source-container">
              <span>📚 근거 자료:</span>
              {msg.sources.map((page, i) => (
                <span key={i} className="source-badge">
                  {page}
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