import React, { useState, useEffect } from 'react';
import Chat from './Chat';
import Login from './Login';
import './App.css';

const API_BASE = 'http://localhost:8080';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [username, setUsername] = useState(localStorage.getItem('username'));

  useEffect(() => {
    if (token) {
      fetch(`${API_BASE}/auth/verify`, {
        headers: { Authorization: `Bearer ${token}` },
      })
        .then(res => res.json())
        .then(data => {
          if (!data.valid) {
            setToken(null);
            setUsername(null);
            localStorage.removeItem('token');
            localStorage.removeItem('username');
          }
        })
        .catch(() => {
          setToken(null);
          setUsername(null);
          localStorage.removeItem('token');
          localStorage.removeItem('username');
        });
    }
  }, [token]);

  const handleLogin = (newToken, newUsername) => {
    setToken(newToken);
    setUsername(newUsername);
    localStorage.setItem('token', newToken);
    localStorage.setItem('username', newUsername);
  };

  const handleLogout = () => {
    setToken(null);
    setUsername(null);
    localStorage.removeItem('token');
    localStorage.removeItem('username');
  };

  if (!token) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <h1>WYF Agent</h1>
        </div>
        <div className="header-right">
          <span className="username">{username}</span>
          <button onClick={handleLogout} className="logout-btn">退出</button>
        </div>
      </header>
      <main className="app-main">
        <Chat token={token} />
      </main>
    </div>
  );
}

export default App;
