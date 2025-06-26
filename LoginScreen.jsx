import React, { useState } from 'react';
import {
  Container, TextField, Button, Typography, Paper, Avatar, Box, Snackbar, Alert
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import cdacLogo from './image/CDAC.jpg';

export default function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [alert, setAlert] = useState({ open: false, message: '', severity: 'error' });

  const handleLogin = async () => {
    if (!username || !password) {
      setAlert({
        open: true,
        message: 'Please enter both username and password.',
        severity: 'warning'
      });
      return;
    }

    try {
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);

      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: formData.toString()
      });

      console.log("Login response:", response);

      if (!response.ok) throw new Error('Login failed');

      const data = await response.json();
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('user_id', username);
      onLogin(username);
    } catch (err) {
      console.error('Login error:', err);
      setAlert({
        open: true,
        message: 'Login failed. Please try again.',
        severity: 'error'
      });
    }
  };

  return (
    <Container maxWidth="xs" sx={{ mt: 8 }}>
      <Paper elevation={6} sx={{ padding: 4, borderRadius: 3 }}>
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <Avatar sx={{ m: 'auto', bgcolor: 'primary.main' }}>
            <LockOutlinedIcon />
          </Avatar>
          <Typography variant="h5" fontWeight="bold" mt={1}>
            Login to CDAC ChatBot
          </Typography>
          <Box sx={{ mt: 1 }}>
            <img
              src={cdacLogo}
              alt="CDAC Logo"
              style={{ width: 80, height: 80, borderRadius: '50%' }}
            />
          </Box>
        </Box>

        <TextField
          fullWidth
          label="Username"
          variant="outlined"
          margin="normal"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />

        <TextField
          fullWidth
          label="Password"
          type="password"
          variant="outlined"
          margin="normal"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleLogin()}
        />

        <Button
          variant="contained"
          color="primary"
          fullWidth
          onClick={handleLogin}
          sx={{ mt: 2 }}
        >
          Login
        </Button>
      </Paper>

      <Snackbar
        open={alert.open}
        autoHideDuration={4000}
        onClose={() => setAlert({ ...alert, open: false })}
      >
        <Alert severity={alert.severity} variant="filled">
          {alert.message}
        </Alert>
      </Snackbar>
    </Container>
  );
}
