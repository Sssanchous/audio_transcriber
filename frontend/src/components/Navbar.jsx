import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Navbar() {
  const { user, logout } = useAuth();
  const location = useLocation();

  const linkClass = (path) =>
    `px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      location.pathname === path
        ? 'bg-indigo-600 text-white'
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`;

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-2">
            <Link to="/" className="text-white font-bold text-lg tracking-tight">
              PM Insights
            </Link>
            {user && (
              <div className="hidden sm:flex items-center gap-1 ml-6">
                <Link to="/" className={linkClass('/')}>
                  Загрузка
                </Link>
                <Link to="/history" className={linkClass('/history')}>
                  История
                </Link>
              </div>
            )}
          </div>
          {user && (
            <div className="flex items-center gap-3">
              <span className="text-gray-400 text-sm hidden sm:inline">
                {user.username}
              </span>
              <button
                onClick={logout}
                className="text-sm text-gray-400 hover:text-white transition-colors cursor-pointer"
              >
                Выйти
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
}
