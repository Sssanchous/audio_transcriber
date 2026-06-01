import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const linkClass = (path) =>
    `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      location.pathname === path ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center h-14 gap-4">
          <Link to="/" className="text-white font-bold text-lg tracking-tight">PM Insights</Link>
          <Link to="/" className={linkClass('/')}>Загрузка</Link>
          <Link to="/history" className={linkClass('/history')}>Архив</Link>
          <Link to="/dashboard" className={linkClass('/dashboard')}>Сводка по проекту</Link>
          <div className="ml-auto flex items-center gap-3">
            {user ? (
              <>
                <span className="text-sm text-gray-400">{user.username || user.email}</span>
                <button
                  type="button"
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
                  className="px-3 py-2 rounded-lg text-sm font-medium text-gray-300 hover:bg-gray-700 hover:text-white"
                >
                  Выйти
                </button>
              </>
            ) : (
              <>
                <Link to="/login" className={linkClass('/login')}>Войти</Link>
                <Link to="/register" className={linkClass('/register')}>Регистрация</Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
