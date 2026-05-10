import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../lib/api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

function StatCard({ label, value, sub, color }) {
  return (
    <div className="bg-gray-800/60 rounded-xl p-4 border border-gray-700/50">
      <p className="text-xs text-gray-400 uppercase tracking-wider mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color || 'text-white'}`}>{value}</p>
      {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
    </div>
  );
}

function SentimentBadge({ label, score }) {
  const cls =
    label === 'positive'
      ? 'bg-green-900/40 text-green-400 border-green-700'
      : label === 'negative'
        ? 'bg-red-900/40 text-red-400 border-red-700'
        : 'bg-gray-800 text-gray-400 border-gray-700';
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs border ${cls}`}>
      {label} ({score})
    </span>
  );
}

function DownloadButton({ recordId, format, label, bgClass }) {
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState('');

  const handleDownload = async () => {
    setDownloading(true);
    setError('');
    try {
      const res = await api.get(`/export/${format}/${recordId}`, {
        responseType: 'blob',
      });
      const contentDisposition = res.headers['content-disposition'] || '';
      let filename = `report.${format === 'pdf' ? 'pdf' : 'xlsx'}`;
      const match = contentDisposition.match(/filename="?([^";\n]+)"?/);
      if (match) filename = match[1];

      const url = window.URL.createObjectURL(res.data);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch {
      setError('Ошибка скачивания');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={handleDownload}
        disabled={downloading}
        className={`px-3 py-1.5 ${bgClass} text-white text-sm rounded-lg transition-colors cursor-pointer disabled:opacity-50`}
      >
        {downloading ? '...' : label}
      </button>
      {error && (
        <span className="absolute top-full left-0 mt-1 text-xs text-red-400 whitespace-nowrap">
          {error}
        </span>
      )}
    </div>
  );
}

export default function ResultPage() {
  const { id } = useParams();
  const [rec, setRec] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchData = () => {
    setLoading(true);
    setError('');
    api
      .get(`/records/${id}`)
      .then((res) => setRec(res.data))
      .catch((err) => setError(err.response?.data?.detail || 'Ошибка загрузки'))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchData();
  }, [id]);

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={fetchData} />;
  if (!rec) return null;

  const tasks = rec.tasks_json || [];
  const qa = rec.qa_json || [];
  const segments = rec.segments || [];
  const analytics = rec.analytics_json || {};
  const timing = rec.timing_json || {};
  const dynamic = analytics.dynamic_analysis || {};
  const taskChanges = dynamic.task_changes || {};
  const fullText = rec.full_text || '';

  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold text-white">{rec.filename}</h2>
          <p className="text-gray-400 text-sm mt-1">
            {rec.project_name && `Проект: ${rec.project_name}`}
            {rec.meeting_date && ` | Дата: ${rec.meeting_date}`}
            {rec.participants && ` | ${rec.participants}`}
          </p>
          <p className="text-gray-500 text-xs mt-1">
            Статус: обработка завершена | Язык: {rec.language || 'unknown'}
          </p>
        </div>
        <div className="flex gap-2 items-start">
          <DownloadButton
            recordId={rec.id}
            format="pdf"
            label="Скачать PDF"
            bgClass="bg-red-600 hover:bg-red-700"
          />
          <DownloadButton
            recordId={rec.id}
            format="excel"
            label="Скачать Excel"
            bgClass="bg-green-600 hover:bg-green-700"
          />
          <Link
            to="/history"
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg transition-colors"
          >
            Назад
          </Link>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard label="Длительность" value={`${rec.duration || 0}с`} />
        <StatCard label="Задач" value={tasks.length} />
        <StatCard label="Q/A пар" value={qa.length} />
        <StatCard
          label="Sentiment"
          value={analytics.avg_sentiment_score ?? 0}
          color={
            (analytics.avg_sentiment_score ?? 0) > 0
              ? 'text-green-400'
              : (analytics.avg_sentiment_score ?? 0) < 0
                ? 'text-red-400'
                : 'text-gray-300'
          }
        />
        <StatCard label="Негатив" value={analytics.negative_ratio ?? 0} />
        <StatCard label="Время" value={`${timing.total || 0}с`} sub={`whisper: ${timing.transcribe || 0}с`} />
      </div>

      {/* Timing */}
      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span>upload: {timing.upload || 0}с</span>
        <span>convert: {timing.convert || 0}с</span>
        <span>transcribe: {timing.transcribe || 0}с</span>
        <span>nlp: {timing.nlp || 0}с</span>
        <span>total: {timing.total || 0}с</span>
      </div>

      {/* Full Transcription Text */}
      <section className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold text-white mb-1">Транскрибация встречи</h3>
        <p className="text-gray-500 text-xs mb-4">
          {rec.filename}
          {rec.meeting_date && ` | ${rec.meeting_date}`}
          {rec.duration ? ` | ${rec.duration}с` : ''}
        </p>
        {fullText ? (
          <div className="bg-gray-800/50 rounded-xl p-5 border border-gray-700/50 max-h-[400px] overflow-y-auto">
            <p className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
              {fullText}
            </p>
          </div>
        ) : (
          <div className="bg-gray-800/30 rounded-xl p-8 border border-gray-700/30 text-center">
            <p className="text-gray-500 text-sm">Транскрибация пока недоступна</p>
          </div>
        )}
      </section>

      {/* Dynamic Analysis */}
      {dynamic.summary && (
        <section className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
          <h3 className="text-lg font-semibold text-white mb-3">Динамический анализ</h3>
          <p className="text-gray-300 text-sm mb-4">{dynamic.summary}</p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
            <StatCard label="Δ задач" value={dynamic.tasks_delta ?? 0} />
            <StatCard label="Δ Q/A" value={dynamic.qa_delta ?? 0} />
            <StatCard label="Δ sentiment" value={dynamic.sentiment_delta ?? 0} />
            <StatCard label="Δ негатив" value={dynamic.negative_ratio_delta ?? 0} />
          </div>

          {(taskChanges.new_tasks || []).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-300 mb-2">
                Новые задачи ({taskChanges.new_count})
              </h4>
              {taskChanges.new_tasks.map((t, i) => (
                <div key={i} className="bg-gray-800/50 rounded-lg p-3 mb-2 border border-gray-700/50">
                  <p className="text-gray-200 text-sm">{t.text}</p>
                  <p className="text-gray-500 text-xs mt-1">
                    {t.responsible || '—'} | {t.deadline || '—'}
                  </p>
                </div>
              ))}
            </div>
          )}

          {(taskChanges.repeated_tasks || []).length > 0 && (
            <div className="mb-4">
              <h4 className="text-sm font-medium text-gray-300 mb-2">
                Повторяющиеся ({taskChanges.repeated_count})
              </h4>
              {taskChanges.repeated_tasks.map((item, i) => (
                <div key={i} className="bg-gray-800/50 rounded-lg p-3 mb-2 border border-gray-700/50">
                  <p className="text-gray-200 text-sm">
                    Текущая: {item.current_task?.text}
                  </p>
                  <p className="text-gray-400 text-xs">
                    Прошлая: {item.previous_task?.text}
                  </p>
                  <p className="text-gray-500 text-xs mt-1">
                    Сходство: {item.similarity}
                  </p>
                </div>
              ))}
            </div>
          )}

          {(taskChanges.potentially_closed_tasks || []).length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-300 mb-2">
                Потенциально закрытые ({taskChanges.potentially_closed_count})
              </h4>
              {taskChanges.potentially_closed_tasks.map((t, i) => (
                <div key={i} className="bg-gray-800/50 rounded-lg p-3 mb-2 border border-gray-700/50">
                  <p className="text-gray-200 text-sm">{t.text}</p>
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Tasks */}
      <section className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold text-white mb-3">Задачи ({tasks.length})</h3>
        {tasks.length === 0 ? (
          <p className="text-gray-500 text-sm">Задачи не найдены</p>
        ) : (
          <div className="space-y-3">
            {tasks.map((t, i) => (
              <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-gray-200">
                  <span className="text-indigo-400 font-medium mr-2">{i + 1}.</span>
                  {t.text}
                </p>
                <div className="flex flex-wrap gap-3 mt-2 text-xs text-gray-400">
                  <span>Ответственный: <b>{t.responsible || '—'}</b></span>
                  <span>Срок: <b>{t.deadline || '—'}</b></span>
                  <span>Conf: {t.confidence}</span>
                  <SentimentBadge label={t.sentiment?.label || 'neutral'} score={t.sentiment?.score ?? 0} />
                  <span className="text-gray-500">{t.timecode}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Q&A */}
      <section className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold text-white mb-3">Вопросы — ответы ({qa.length})</h3>
        {qa.length === 0 ? (
          <p className="text-gray-500 text-sm">Q/A не найдены</p>
        ) : (
          <div className="space-y-3">
            {qa.map((q, i) => (
              <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                <p className="text-gray-200 text-sm">
                  <span className="text-indigo-400 font-bold mr-1">В:</span> {q.question}{' '}
                  <span className="text-gray-500 text-xs">{q.question_timecode}</span>
                </p>
                <p className="text-gray-300 text-sm mt-1">
                  <span className="text-green-400 font-bold mr-1">О:</span> {q.answer}{' '}
                  <span className="text-gray-500 text-xs">{q.answer_timecode}</span>
                </p>
                <p className="text-gray-500 text-xs mt-1">Conf: {q.confidence}</p>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Transcription segments */}
      <section className="bg-gray-900 rounded-2xl p-6 border border-gray-800">
        <h3 className="text-lg font-semibold text-white mb-3">
          Сегменты транскрипции ({segments.length})
        </h3>
        {segments.length === 0 ? (
          <p className="text-gray-500 text-sm">Сегменты не найдены</p>
        ) : (
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2">
            {segments.map((seg, i) => {
              const borderColor =
                seg.sentiment_label === 'positive'
                  ? 'border-l-green-500'
                  : seg.sentiment_label === 'negative'
                    ? 'border-l-red-500'
                    : 'border-l-gray-600';
              return (
                <div
                  key={i}
                  className={`border-l-4 ${borderColor} bg-gray-800/40 rounded-r-lg px-4 py-2`}
                >
                  <div className="flex justify-between items-start gap-2">
                    <p className="text-gray-200 text-sm flex-1">{seg.text}</p>
                    <SentimentBadge label={seg.sentiment_label} score={seg.sentiment_score} />
                  </div>
                  <p className="text-gray-500 text-xs mt-1">
                    {seg.timecode} | {seg.predicted_label} (conf: {seg.prediction_confidence}, src: {seg.prediction_source})
                  </p>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
