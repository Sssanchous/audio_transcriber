import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setError('');
    if (password !== confirmPassword) {
      setError('Пароли не совпадают.');
      return;
    }
    setLoading(true);
    try {
      await register({ email: email.trim(), username: username.trim(), password, fullName: fullName.trim() });
      navigate('/upload');
    } catch (err) {
      setError(err.response?.data?.detail || 'Не удалось зарегистрироваться.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center px-4">
      <form onSubmit={handleSubmit} className="w-full max-w-md bg-gray-900 border border-gray-800 rounded-lg p-7 space-y-5">
        <div>
          <h1 className="text-2xl font-bold text-white">PM Insights</h1>
          <p className="text-sm text-gray-400 mt-1">Регистрация пользователя</p>
        </div>

        {error && <div className="text-sm text-red-300 bg-red-950/40 border border-red-800 rounded-md p-3">{error}</div>}

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Полное имя</span>
          <input
            value={fullName}
            onChange={(event) => setFullName(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            autoComplete="name"
          />
        </label>

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Email</span>
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            required
            autoComplete="email"
          />
        </label>

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Имя пользователя</span>
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            required
            minLength={3}
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
            minLength={6}
            autoComplete="new-password"
          />
        </label>

        <label className="block">
          <span className="block text-sm text-gray-300 mb-1.5">Повторите пароль</span>
          <input
            type="password"
            value={confirmPassword}
            onChange={(event) => setConfirmPassword(event.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-white"
            required
            minLength={6}
            autoComplete="new-password"
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white font-medium rounded-md py-2.5"
        >
          {loading ? 'Регистрация...' : 'Зарегистрироваться'}
        </button>

        <p className="text-sm text-gray-400 text-center">
          Уже есть аккаунт? <Link to="/login" className="text-indigo-300 hover:text-indigo-200">Войти</Link>
        </p>
      </form>
    </div>
  );
}
