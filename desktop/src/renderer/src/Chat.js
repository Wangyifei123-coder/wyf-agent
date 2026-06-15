import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Paperclip, Image, Loader2 } from 'lucide-react';
import './Chat.css';

const API_BASE = 'http://localhost:8080';

function Chat({ token }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [images, setImages] = useState([]);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && images.length === 0) return;
    if (isLoading) return;

    const userMessage = {
      role: 'user',
      content: input,
      images: images.map(img => img.base64),
      timestamp: new Date().toLocaleTimeString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setImages([]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          message: input,
          images: userMessage.images.length > 0 ? userMessage.images : null,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = {
        role: 'assistant',
        content: '',
        intent: '',
        sources: [],
        timestamp: new Date().toLocaleTimeString(),
      };

      setMessages(prev => [...prev, assistantMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value);
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.type === 'chunk') {
                assistantMessage.content += data.chunk;
                setMessages(prev => {
                  const newMessages = [...prev];
                  newMessages[newMessages.length - 1] = { ...assistantMessage };
                  return newMessages;
                });
              } else if (data.type === 'intent') {
                assistantMessage.intent = data.intent;
              } else if (data.type === 'done') {
                assistantMessage.sources = data.sources || [];
              }
            } catch (e) {}
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: '抱歉，请求失败了。', timestamp: new Date().toLocaleTimeString() },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = reader.result.split(',')[1];
        setImages(prev => [...prev, { name: file.name, base64, preview: reader.result }]);
      };
      reader.readAsDataURL(file);
    });
  };

  const removeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h2>WYF Agent</h2>
            <p>有什么可以帮你的？</p>
          </div>
        )}
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="message-header">
              <span className="role">{msg.role === 'user' ? '你' : 'AI'}</span>
              <span className="time">{msg.timestamp}</span>
              {msg.intent && <span className="intent">{msg.intent}</span>}
            </div>
            <div className="message-content">
              {msg.images && msg.images.length > 0 && (
                <div className="message-images">
                  {msg.images.map((img, i) => (
                    <img key={i} src={`data:image/png;base64,${img}`} alt="" />
                  ))}
                </div>
              )}
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="message-sources">
                来源: {msg.sources.join(', ')}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="message assistant">
            <div className="message-content loading">
              <Loader2 className="spinner" />
              <span>思考中...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {images.length > 0 && (
        <div className="image-preview">
          {images.map((img, index) => (
            <div key={index} className="preview-item">
              <img src={img.preview} alt={img.name} />
              <button onClick={() => removeImage(index)}>×</button>
            </div>
          ))}
        </div>
      )}

      <div className="chat-input">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleImageUpload}
          accept="image/*"
          multiple
          style={{ display: 'none' }}
        />
        <button className="icon-btn" onClick={() => fileInputRef.current?.click()}>
          <Image size={20} />
        </button>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="输入消息..."
          rows={1}
        />
        <button className="send-btn" onClick={handleSend} disabled={isLoading}>
          <Send size={20} />
        </button>
      </div>
    </div>
  );
}

export default Chat;
