import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

export default function HistoryPage() {
  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchRecords = () => {
    setLoading(true);
    setError('');
    api
      .get('/records')
      .then((res) => setRecords(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Ошибка загрузки'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  const handleDelete = async (id) => {
    if (!confirm('Удалить эту запись?')) return;
    try {
      await api.delete(`/records/${id}`);
      setRecords((prev) => prev.filter((r) => r.id !== id));
    } catch {
      alert('Ошибка удаления');
    }
  };

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={fetchRecords} />;

  return (
    <div className="max-w-6xl mx-auto mt-8">
      <h2 className="text-2xl font-bold text-white mb-6">История транскрипций</h2>
      {records.length === 0 ? (
        <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800 text-center">
          <p className="text-gray-400">Пока нет транскрипций</p>
          <Link
            to="/"
            className="inline-block mt-4 text-indigo-400 hover:text-indigo-300"
          >
            Загрузить аудио
          </Link>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-800 text-left text-xs uppercase tracking-wider text-gray-400">
                  <th className="px-4 py-3">Файл</th>
                  <th className="px-4 py-3">Проект</th>
                  <th className="px-4 py-3 hidden sm:table-cell">Дата</th>
                  <th className="px-4 py-3 hidden md:table-cell">Длит.</th>
                  <th className="px-4 py-3 hidden md:table-cell">Задачи</th>
                  <th className="px-4 py-3 hidden lg:table-cell">Q/A</th>
                  <th className="px-4 py-3 hidden lg:table-cell">Sentiment</th>
                  <th className="px-4 py-3 w-16"></th>
                </tr>
              </thead>
              <tbody>
                {records.map((rec) => {
                  const analytics = rec.analytics_json || {};
                  return (
                    <tr
                      key={rec.id}
                      className="border-b border-gray-800/50 hover:bg-gray-800/40 transition-colors"
                    >
                      <td className="px-4 py-3">
                        <Link
                          to={`/result/${rec.id}`}
                          className="text-indigo-400 hover:text-indigo-300 font-medium"
                        >
                          {rec.filename}
                        </Link>
                      </td>
                      <td className="px-4 py-3 text-gray-300 text-sm">
                        {rec.project_name || '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-400 text-sm hidden sm:table-cell">
                        {rec.meeting_date || '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-400 text-sm hidden md:table-cell">
                        {rec.duration ? `${rec.duration}с` : '—'}
                      </td>
                      <td className="px-4 py-3 text-gray-400 text-sm hidden md:table-cell">
                        {(rec.tasks_json || []).length}
                      </td>
                      <td className="px-4 py-3 text-gray-400 text-sm hidden lg:table-cell">
                        {(rec.qa_json || []).length}
                      </td>
                      <td className="px-4 py-3 text-sm hidden lg:table-cell">
                        <span
                          className={
                            analytics.avg_sentiment_score > 0
                              ? 'text-green-400'
                              : analytics.avg_sentiment_score < 0
                                ? 'text-red-400'
                                : 'text-gray-400'
                          }
                        >
                          {analytics.avg_sentiment_score ?? 0}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => handleDelete(rec.id)}
                          className="text-gray-500 hover:text-red-400 transition-colors cursor-pointer"
                          title="Удалить"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
