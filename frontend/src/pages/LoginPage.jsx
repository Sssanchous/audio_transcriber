import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [loginValue, setLoginValue] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    setLoading(true);
    try {
      await login(loginValue.trim(), password);
      navigate('/upload');
    } catch (err) {
      setError(err.response?.data?.detail || 'Не удалось войти. Проверьте логин и пароль.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-gray-900 border border-gray-800 rounded-lg p-7 space-y-5">
        <div>
          <h1 className="text-2xl font-bold text-white">PM Insights</h1>
          <p className="text-sm text-gray-400 mt-1">Вход в систему</p>
        </div>

        {error && <div className="text-sm text-red-300 bg-red-950/40 border border-red-800 rounded-md p-3">{error}</div>}

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Email или имя пользователя</span>
          <input
            value={loginValue}
            onChange={(event) => setLoginValue(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            required
            autoComplete="username"
          />
        </label>

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Пароль</span>
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            required
            autoComplete="current-password"
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white font-medium rounded-md py-2.5"
        >
          {loading ? 'Вход...' : 'Войти'}
        </button>

        <p className="text-sm text-gray-400 text-center">
          Нет аккаунта? <Link to="/register" className="text-indigo-300 hover:text-indigo-200">Зарегистрироваться</Link>
        </p>
      </form>
    </div>
  );
}
