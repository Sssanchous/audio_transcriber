/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import ErrorMessage from '../components/ErrorMessage';

function HistorySkeleton() {
  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6 animate-pulse">
      <div className="space-y-2">
        <div className="h-7 w-48 bg-gray-800 rounded" />
        <div className="h-4 w-80 bg-gray-800 rounded" />
      </div>
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <div className="h-10 bg-gray-800 rounded-lg" />
      </div>
      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-800">
          <div className="h-4 w-full bg-gray-800 rounded" />
        </div>
        {Array.from({ length: 6 }).map((_, index) => (
          <div key={index} className="px-4 py-4 border-b border-gray-800/50 flex gap-4">
            <div className="h-4 flex-1 bg-gray-800 rounded" />
            <div className="h-4 w-24 bg-gray-800 rounded" />
            <div className="h-4 w-24 bg-gray-800 rounded" />
            <div className="h-4 w-20 bg-gray-800 rounded" />
          </div>
        ))}
      </div>
    </div>
  );
}

function filenameFromDisposition(disposition, fallback) {
  const match = disposition?.match(/filename="?([^";]+)"?/i);
  return match ? decodeURIComponent(match[1]) : fallback;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

const PAGE_SIZE = 10;

export default function HistoryPage() {
  const [meetings, setMeetings] = useState([]);
  const [filter, setFilter] = useState('');
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloadError, setDownloadError] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [deletingId, setDeletingId] = useState('');

  const load = () => {
    setLoading(true);
    setError('');
    api.get('/meetings')
      .then((response) => setMeetings(response.data))
      .catch((err) => setError(err.response?.data?.detail || 'Ошибка загрузки архива'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const downloadReport = async (meetingId, format) => {
    setDownloadError('');
    try {
      const response = await api.get(`/meetings/${meetingId}/export/${format}`, { responseType: 'blob' });
      const fallback = `pm_insights_${meetingId}.${format}`;
      downloadBlob(response.data, filenameFromDisposition(response.headers['content-disposition'], fallback));
    } catch (err) {
      setDownloadError(err.response?.data?.detail || 'Не удалось скачать отчёт.');
    }
  };

  const deleteMeeting = async (meeting) => {
    const meetingId = meeting.meeting_id;
    if (!window.confirm('Удалить встречу? Это действие нельзя отменить.')) return;
    setDeleteError('');
    setDeletingId(meetingId);
    try {
      await api.delete(`/meetings/${meetingId}`);
      setMeetings((current) => current.filter((item) => item.meeting_id !== meetingId));
    } catch (err) {
      setDeleteError(err.response?.data?.detail || 'Не удалось удалить встречу.');
    } finally {
      setDeletingId('');
    }
  };

  if (loading) return <HistorySkeleton />;
  if (error) return <ErrorMessage message={error} onRetry={load} />;

  const filteredMeetings = meetings.filter((meeting) => {
    const haystack = `${meeting.meeting_title || ''} ${meeting.project_name || ''} ${meeting.original_filename || ''}`.toLowerCase();
    return haystack.includes(filter.toLowerCase());
  });
  const totalPages = Math.max(1, Math.ceil(filteredMeetings.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pagedMeetings = filteredMeetings.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white">Архив встреч</h2>
        <p className="text-gray-500 text-sm mt-1">Список встреч, результатов и отчётов пользователя.</p>
      </div>

      {downloadError && (
        <div className="bg-red-950/20 border border-red-900/60 rounded-xl p-3 text-sm text-red-200">
          {downloadError}
        </div>
      )}
      {deleteError && (
        <div className="bg-red-950/20 border border-red-900/60 rounded-xl p-3 text-sm text-red-200">
          {deleteError}
        </div>
      )}

      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <input
          value={filter}
          onChange={(event) => { setFilter(event.target.value); setPage(1); }}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white"
          placeholder="Фильтр по названию встречи, проекту или файлу"
        />
      </div>

      <div className="bg-gray-900 rounded-xl border border-gray-800 overflow-hidden">
        {filteredMeetings.length === 0 ? (
          <div className="p-8 text-center text-gray-400">Пока нет загруженных встреч</div>
        ) : (
          <table className="w-full text-sm">
            <thead className="text-gray-400 border-b border-gray-800">
              <tr>
                <th className="text-left px-4 py-3">Встреча</th>
                <th className="text-left px-4 py-3">Проект</th>
                <th className="text-left px-4 py-3">Дата встречи</th>
                <th className="text-left px-4 py-3">Дата загрузки</th>
                <th className="text-left px-4 py-3">Статус</th>
                <th className="text-left px-4 py-3">Действия</th>
              </tr>
            </thead>
            <tbody>
              {pagedMeetings.map((meeting) => (
                <tr key={meeting.meeting_id} className="border-b border-gray-800/50">
                  <td className="px-4 py-3">
                    <Link className="text-indigo-400 hover:text-indigo-300" to={`/result/${meeting.meeting_id}`}>
                      {meeting.meeting_title || meeting.original_filename}
                    </Link>
                    <p className="text-xs text-gray-500">{meeting.original_filename}</p>
                  </td>
                  <td className="px-4 py-3 text-gray-300">{meeting.project_name}</td>
                  <td className="px-4 py-3 text-gray-400">{meeting.meeting_date || '-'}</td>
                  <td className="px-4 py-3 text-gray-400">{meeting.upload_date?.slice(0, 10) || '-'}</td>
                  <td className="px-4 py-3 text-gray-400">{meeting.processing_status}</td>
                  <td className="px-4 py-3">
                    <div className="flex flex-wrap gap-2">
                      <Link className="text-indigo-400 hover:text-indigo-300" to={`/result/${meeting.meeting_id}`}>Открыть</Link>
                      <button type="button" className="text-gray-300 hover:text-white" onClick={() => downloadReport(meeting.meeting_id, 'pdf')}>PDF</button>
                      <button type="button" className="text-gray-300 hover:text-white" onClick={() => downloadReport(meeting.meeting_id, 'xlsx')}>Excel</button>
                      <button type="button" className="text-gray-300 hover:text-white" onClick={() => downloadReport(meeting.meeting_id, 'docx')}>Word</button>
                      <button
                        type="button"
                        className="text-red-300 hover:text-red-100 disabled:opacity-60"
                        disabled={deletingId === meeting.meeting_id}
                        onClick={() => deleteMeeting(meeting)}
                      >
                        {deletingId === meeting.meeting_id ? 'Удаление...' : 'Удалить'}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-800 text-sm text-gray-400">
            <button
              type="button"
              disabled={currentPage <= 1}
              onClick={() => setPage((value) => Math.max(1, value - 1))}
              className="px-3 py-1.5 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Назад
            </button>
            <span>Страница {currentPage} из {totalPages}</span>
            <button
              type="button"
              disabled={currentPage >= totalPages}
              onClick={() => setPage((value) => Math.min(totalPages, value + 1))}
              className="px-3 py-1.5 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Далее
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
