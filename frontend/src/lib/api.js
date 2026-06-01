import axios from 'axios';

const api = axios.create({ baseURL: '/api' });

export const TOKEN_KEY = 'pm_insights_token';

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || localStorage.getItem('token');
}

export function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem('token', token);
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem('token');
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken();
      const path = window.location.pathname;
      if (!['/login', '/register'].includes(path)) {
        window.location.assign('/login');
      }
    }
    return Promise.reject(error);
  },
);

export default api;
