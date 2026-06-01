/* eslint-disable react-hooks/set-state-in-effect */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import api from '../lib/api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

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

export default function HistoryPage() {
  const [meetings, setMeetings] = useState([]);
  const [filter, setFilter] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [downloadError, setDownloadError] = useState('');

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

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={load} />;

  const filteredMeetings = meetings.filter((meeting) => {
    const haystack = `${meeting.meeting_title || ''} ${meeting.project_name || ''} ${meeting.original_filename || ''}`.toLowerCase();
    return haystack.includes(filter.toLowerCase());
  });

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

      <div className="bg-gray-900 rounded-xl border border-gray-800 p-4">
        <input
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
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
              {filteredMeetings.map((meeting) => (
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
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

