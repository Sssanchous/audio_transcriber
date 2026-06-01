/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import api, { clearToken, getToken, setToken } from '../lib/api';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(Boolean(getToken()));

  useEffect(() => {
    if (!getToken()) {
      return;
    }

    api
      .get('/auth/me')
      .then((response) => setUser(response.data.user || response.data))
      .catch(() => {
        clearToken();
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = async (loginValue, password) => {
    const payload = loginValue.includes('@')
      ? { email: loginValue, password }
      : { username: loginValue, password };
    const response = await api.post('/auth/login', payload);
    setToken(response.data.access_token || response.data.token);
    setUser(response.data.user);
    return response.data;
  };

  const register = async ({ email, username, password, fullName = '' }) => {
    const response = await api.post('/auth/register', {
      email,
      username,
      password,
      full_name: fullName,
    });
    setToken(response.data.access_token || response.data.token);
    setUser(response.data.user);
    return response.data;
  };

  const logout = () => {
    clearToken();
    setUser(null);
  };

  const value = useMemo(
    () => ({ user, loading, isAuthenticated: Boolean(user), login, register, logout }),
    [user, loading],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
