interface InputAreaProps {
  input: string;
  setInput: (val: string) => void;
  onSend: () => void;
  loading: boolean;
}

export default function InputArea({ input, setInput, onSend, loading }: InputAreaProps) {
  return (
    <div className="input-area">
      <input 
        type="text" 
        className="chat-input"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyPress={(e) => e.key === 'Enter' && !loading && onSend()}
        placeholder="질문을 입력하세요..."
        disabled={loading}
      />
      <button className="btn-send" onClick={onSend} disabled={loading}>
        전송
      </button>
    </div>
  );
}