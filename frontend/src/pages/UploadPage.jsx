import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../lib/api';
import Spinner from '../components/Spinner';

export default function UploadPage() {
  const navigate = useNavigate();
  const fileRef = useRef(null);
  const [file, setFile] = useState(null);
  const [projectName, setProjectName] = useState('');
  const [participants, setParticipants] = useState('');
  const [meetingDate, setMeetingDate] = useState('');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setError('');
    setUploading(true);
    setProgress('Загрузка и обработка аудио...');

    const form = new FormData();
    form.append('file', file);
    form.append('project_name', projectName);
    form.append('participants', participants);
    if (meetingDate) form.append('meeting_date', meetingDate);

    try {
      const res = await api.post('/upload', form, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000,
      });
      navigate(`/result/${res.data.record_id}`);
    } catch (err) {
      setError(err.response?.data?.detail || 'Ошибка загрузки файла');
    } finally {
      setUploading(false);
      setProgress('');
    }
  };

  const dragOver = (e) => e.preventDefault();

  const drop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  };

  if (uploading) {
    return (
      <div className="max-w-2xl mx-auto mt-12">
        <div className="bg-gray-900 rounded-2xl p-8 border border-gray-800">
          <Spinner text={progress} />
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto mt-8">
      <h2 className="text-2xl font-bold text-white mb-6">Загрузить аудио</h2>
      <form
        onSubmit={handleSubmit}
        className="bg-gray-900 rounded-2xl p-8 border border-gray-800 space-y-6"
      >
        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
            {error}
          </div>
        )}

        <div
          onDragOver={dragOver}
          onDrop={drop}
          onClick={() => fileRef.current?.click()}
          className="border-2 border-dashed border-gray-700 rounded-xl p-8 text-center cursor-pointer hover:border-indigo-500 transition-colors"
        >
          <input
            ref={fileRef}
            type="file"
            accept=".mp3,.wav,.m4a,.opus"
            className="hidden"
            onChange={(e) => setFile(e.target.files[0])}
          />
          {file ? (
            <div>
              <p className="text-white font-medium">{file.name}</p>
              <p className="text-gray-400 text-sm mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} МБ
              </p>
            </div>
          ) : (
            <div>
              <svg className="w-10 h-10 mx-auto text-gray-500 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-gray-400">
                Перетащите файл или{' '}
                <span className="text-indigo-400">нажмите для выбора</span>
              </p>
              <p className="text-gray-500 text-xs mt-1">MP3, WAV, M4A, OPUS</p>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">
              Проект
            </label>
            <input
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              placeholder="PM Insights"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">
              Дата встречи
            </label>
            <input
              type="date"
              value={meetingDate}
              onChange={(e) => setMeetingDate(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1.5">
            Участники
          </label>
          <input
            type="text"
            value={participants}
            onChange={(e) => setParticipants(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            placeholder="Иван, Анна, Сергей"
          />
        </div>

        <button
          type="submit"
          disabled={!file}
          className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white font-medium py-3 rounded-lg transition-colors cursor-pointer"
        >
          Загрузить и обработать
        </button>
      </form>
    </div>
  );
}
