import React, { useState, useEffect, useRef } from 'react';
import {
  AppBar, Toolbar, Typography, Container, TextField, Button, Box, IconButton,
  Snackbar, Alert, FormControlLabel, Switch
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import cdacLogo from './image/CDAC.jpg';
import LoginScreen from './LoginScreen';

function MessageBubble({ message, darkTheme }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: message.sender === 'user' ? 'row-reverse' : 'row', marginBottom: 1 }}>
      <Box sx={{
        backgroundColor: message.sender === 'user' ? 'rgb(25, 118, 210)' : darkTheme ? '#999' : '#fff',
        color: message.sender === 'user' ? '#ddd' : darkTheme ? '#000' : 'rgb(25, 118, 210)',
        padding: 2, borderRadius: 2, maxWidth: '80%', wordWrap: 'break-word',
        boxShadow: darkTheme ? 'none' : '0px 3px 6px #00000029'
      }}>
        <Typography variant="body1">{message.text}</Typography>
      </Box>
    </Box>
  );
}

function ChatApp({ username, onLogout }) {
  const [messages, setMessages] = useState([{ sender: 'bot', text: 'Welcome to CDAC-ChatBot application.' }]);
  const [inputValue, setInputValue] = useState('');
  const [isBotTyping, setIsBotTyping] = useState(false);
  const [darkTheme, setDarkTheme] = useState(true);
  const [alertOpen, setAlertOpen] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');
  const [useHindi, setUseHindi] = useState(false);
  const fileInputRef = useRef(null);
  const chatContainerRef = useRef(null);
  const token = localStorage.getItem('access_token');

  useEffect(() => {
    document.getElementById('chat-input')?.focus();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current)
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
  }, [messages]);

  const handleError = async (res) => {
    if (res.status === 401) {
      localStorage.clear();
      onLogout();
    }
    const err = await res.json();
    throw new Error(err.detail || 'Error occurred');
  };

  const sendQuery = async (endpoint) => {
    if (inputValue.trim() === '') return;
    const newMessages = [...messages, { sender: 'user', text: inputValue }];
    setMessages(newMessages);
    setInputValue('');
    setIsBotTyping(true);
    try {
      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ question: inputValue })
      });
      if (!response.ok) await handleError(response);
      const data = await response.json();
      setMessages([...newMessages, { sender: 'bot', text: data.answer }]);
    } catch (err) {
      setMessages([...newMessages, { sender: 'bot', text: err.message }]);
    } finally {
      setIsBotTyping(false);
    }
  };

  const handleUploadDocument = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData
      });
      if (!res.ok) await handleError(res);
      const result = await res.json();
      setAlertMessage(result.message || 'Uploaded!');
    } catch (err) {
      setAlertMessage(err.message || 'Upload failed.');
    }
    setAlertOpen(true);
  };

  const handleDownloadExcel = () => {
    const worksheet = XLSX.utils.json_to_sheet(messages);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, 'ChatOutput');
    const excelBuffer = XLSX.write(workbook, { bookType: 'xlsx', type: 'array' });
    const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
    saveAs(blob, 'ChatOutput.xlsx');
  };

  const handleToggleLanguage = async () => {
    setUseHindi(prev => !prev);
    try {
      const res = await fetch('http://localhost:8000/toggle-language', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ use_hindi: !useHindi })
      });
      if (!res.ok) await handleError(res);
      const data = await res.json();
      setAlertMessage(data.message || 'Language toggled.');
    } catch (err) {
      setAlertMessage(err.message || 'Error toggling language.');
    }
    setAlertOpen(true);
  };

  const theme = createTheme({ palette: { mode: darkTheme ? 'dark' : 'light' } });

  return (
    <ThemeProvider theme={theme}>
      <div style={{ height: '100vh', background: darkTheme ? '#111' : '#fff', color: darkTheme ? '#fff' : '#000' }}>
        <AppBar position="static" style={{ background: darkTheme ? '#222' : '#1976d2' }}>
          <Toolbar>
            <img src={cdacLogo} alt="CDAC Logo" style={{ width: 50, height: 50, borderRadius: '50%' }} />
            <Typography variant="h6" sx={{ flex: 1, textAlign: 'center' }}>CDAC-CHATBOT</Typography>
            <IconButton onClick={() => setDarkTheme(!darkTheme)} color="inherit"><Brightness4Icon /></IconButton>
            <Button color="inherit" onClick={() => { localStorage.clear(); onLogout(); }}>Logout</Button>
          </Toolbar>
        </AppBar>

        <Container sx={{ padding: 2, height: 580, display: 'flex', flexDirection: 'column', mt: 5 }}>
          <Box ref={chatContainerRef} sx={{ flexGrow: 1, overflowY: 'auto', padding: 2, borderRadius: '10px', mb: 2, background: darkTheme ? 'rgb(34, 34, 34)' : '#fff' }}>
            {messages.map((msg, idx) => <MessageBubble key={idx} message={msg} darkTheme={darkTheme} />)}
            {isBotTyping && <Typography variant="body1">Bot typing...</Typography>}
          </Box>

          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <TextField
              id="chat-input"
              label="Type your message..."
              variant="outlined"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && sendQuery('/query/book')}
              fullWidth sx={{ mb: 2 }}
            />
            <FormControlLabel control={<Switch checked={useHindi} onChange={handleToggleLanguage} />} label="Hindi Response" />
            <Box sx={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
              <Button variant="contained" onClick={() => sendQuery('/query/book')} sx={{ m: 1 }}>Ask Book</Button>
              <Button variant="contained" onClick={() => sendQuery('/query/documents')} sx={{ m: 1 }}>Ask Docs</Button>
              <Button variant="outlined" startIcon={<CloudUploadIcon />} onClick={() => fileInputRef.current.click()} sx={{ m: 1 }}>Upload</Button>
              <Button variant="outlined" endIcon={<CloudDownloadIcon />} onClick={handleDownloadExcel} sx={{ m: 1 }}>Download</Button>
            </Box>
            <input type="file" style={{ display: 'none' }} onChange={handleUploadDocument} ref={fileInputRef} />
          </Box>

          <Snackbar open={alertOpen} autoHideDuration={3000} onClose={() => setAlertOpen(false)}>
            <Alert severity="info">{alertMessage}</Alert>
          </Snackbar>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default function ChatUI() {
  const [username, setUsername] = useState(localStorage.getItem('user_id'));

  const handleLogin = (username) => setUsername(username);
  const handleLogout = () => setUsername(null);

  return username
    ? <ChatApp username={username} onLogout={handleLogout} />
    : <LoginScreen onLogin={handleLogin} />;
}
