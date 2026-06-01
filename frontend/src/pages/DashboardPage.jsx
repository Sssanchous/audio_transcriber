/* eslint-disable react-hooks/set-state-in-effect */
/* eslint-disable react-hooks/exhaustive-deps */
import { useEffect, useMemo, useState } from 'react';
import api from '../lib/api';
import Spinner from '../components/Spinner';
import ErrorMessage from '../components/ErrorMessage';

function EmptyState({ children }) {
  return <div className="bg-white border border-gray-200 rounded-lg p-6 text-sm text-gray-500">{children}</div>;
}

function StatCard({ label, value }) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <p className="text-gray-500 text-xs">{label}</p>
      <p className="text-gray-950 text-2xl font-semibold mt-1">{value ?? 0}</p>
    </div>
  );
}

function ChartCard({ title, description, children }) {
  return (
    <section className="bg-white border border-gray-200 rounded-xl p-5 space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-gray-950">{title}</h3>
        <p className="text-sm text-gray-500 mt-1">{description}</p>
      </div>
      {children}
    </section>
  );
}

function LineChart({ data }) {
  if (!data.length) return <EmptyState>Нет данных по выбранному проекту.</EmptyState>;
  const width = 640;
  const height = 220;
  const values = data.map((item) => Number(item.average_sentiment || 0));
  const min = Math.min(...values, -1);
  const max = Math.max(...values, 1);
  const range = max - min || 1;
  const step = data.length > 1 ? width / (data.length - 1) : width / 2;
  const points = data.map((item, index) => ({
    ...item,
    x: data.length > 1 ? index * step : width / 2,
    y: height - ((Number(item.average_sentiment || 0) - min) / range) * height,
  }));
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width + 40} ${height + 54}`} className="min-w-[520px] w-full h-72">
        {[0, 0.25, 0.5, 0.75, 1].map((line) => (
          <line key={line} x1="0" x2={width} y1={height * line} y2={height * line} stroke="#e5e7eb" />
        ))}
        <path d={path} fill="none" stroke="#111827" strokeWidth="3" />
        {points.map((point) => (
          <g key={`${point.meeting_id}-${point.date}`} transform={`translate(${point.x} ${point.y})`}>
            <circle r="5" fill="#111827" />
            <title>{`${point.meeting_title || 'Встреча'}\n${point.date}\nСредний тон: ${point.average_sentiment}`}</title>
          </g>
        ))}
        {points.map((point) => (
          <text key={`${point.meeting_id}-${point.date}-label`} x={point.x} y={height + 24} fontSize="11" textAnchor="middle" fill="#4b5563">
            {point.date}
          </text>
        ))}
      </svg>
    </div>
  );
}

function BarChart({ data }) {
  if (!data.length) return <EmptyState>Нет данных по выбранному проекту.</EmptyState>;
  const max = Math.max(...data.map((item) => Number(item.tasks_count || 0)), 1);
  return (
    <div className="space-y-3">
      {data.map((item) => {
        const value = Number(item.tasks_count || 0);
        return (
          <div key={`${item.meeting_id}-${item.date}`} className="space-y-1">
            <div className="flex justify-between gap-4 text-xs text-gray-500">
              <span>{item.date} · {item.meeting_title || 'Встреча'}</span>
              <span>{value}</span>
            </div>
            <div className="h-7 bg-gray-100 border border-gray-200 rounded">
              <div className="h-full bg-gray-950 rounded" style={{ width: `${Math.max(4, (value / max) * 100)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function WordCloud({ words }) {
  if (!words.length) return <EmptyState>Нет аспектов по выбранному проекту.</EmptyState>;
  const max = Math.max(...words.map((item) => Number(item.value || 0)), 1);
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-3 leading-none">
      {words.slice(0, 30).map((item) => {
        const size = 14 + (Number(item.value || 0) / max) * 22;
        return (
          <span
            key={item.text}
            className="text-gray-950"
            style={{ fontSize: `${size}px`, fontWeight: item.value > max / 2 ? 700 : 500 }}
            title={`${item.text}: ${item.value}`}
          >
            {item.text}
          </span>
        );
      })}
    </div>
  );
}

export default function DashboardPage() {
  const [dashboard, setDashboard] = useState(null);
  const [project, setProject] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = (projectKey = project) => {
    setLoading(true);
    setError('');
    api.get('/dashboard', { params: { project: projectKey || undefined } })
      .then((response) => {
        setDashboard(response.data);
        if (!projectKey && response.data.selected_project) {
          setProject(response.data.selected_project);
        }
      })
      .catch((err) => setError(err.response?.data?.detail || 'Ошибка загрузки дашборда'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(''); }, []);

  const projects = useMemo(() => dashboard?.projects || [], [dashboard]);
  const summary = dashboard?.summary || {};

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={() => load(project)} />;
  if (!dashboard) return null;

  const handleProjectChange = (event) => {
    const value = event.target.value;
    setProject(value);
    load(value);
  };

  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-950">Сводка по проекту</h2>
        <p className="text-gray-500 text-sm mt-2">
          Динамика настроения, количества задач и обсуждаемых аспектов по выбранному проекту.
        </p>
      </div>

      <section className="bg-white border border-gray-200 rounded-xl p-5">
        <label className="text-sm text-gray-700 block max-w-xl">
          <span className="block mb-1 font-medium">Проект</span>
          <select
            value={project}
            onChange={handleProjectChange}
            className="w-full border border-gray-300 rounded-lg px-3 py-2 bg-white text-gray-950"
          >
            <option value="">Все проекты</option>
            {projects.map((item) => (
              <option key={item.project_key} value={item.project_key}>{item.project_name}</option>
            ))}
          </select>
        </label>
      </section>

      {!projects.length && <EmptyState>Пока нет обработанных встреч для построения сводки.</EmptyState>}

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
        <StatCard label="Встречи" value={summary.meetings_count} />
        <StatCard label="Завершено" value={summary.completed_count} />
        <StatCard label="Задачи" value={summary.tasks_count} />
        <StatCard label="Вопросы" value={summary.questions_count} />
        <StatCard label="Ответы" value={summary.answers_count} />
        <StatCard label="Решения" value={summary.decisions_count} />
        <StatCard label="Ответственные" value={summary.responsibles_count} />
        <StatCard label="Дедлайны" value={summary.deadlines_count} />
        <StatCard label="Средний тон" value={summary.average_sentiment} />
        <StatCard label="Обработка, сек." value={summary.average_processing_seconds ?? '-'} />
        <StatCard label="Оценка 1 часа, мин." value={summary.average_estimated_1h_minutes ?? '-'} />
      </div>

      <ChartCard title="Динамика настроения команды" description="Средняя тональность встреч по датам.">
        <LineChart data={dashboard.sentiment_trend || []} />
      </ChartCard>

      <ChartCard title="Динамика количества поставленных задач" description="Количество выделенных задач/action items по встречам.">
        <BarChart data={dashboard.task_trend || []} />
      </ChartCard>

      <ChartCard title="Облако обсуждаемых аспектов" description="Самые обсуждаемые аспекты и темы по выбранному проекту.">
        <WordCloud words={dashboard.aspect_word_cloud || []} />
      </ChartCard>

      <details className="bg-white border border-gray-200 rounded-xl p-5">
        <summary className="cursor-pointer text-sm font-medium text-gray-950">Показать технические показатели</summary>
        <div className="grid md:grid-cols-3 gap-3 mt-4 text-sm">
          <StatCard label="Длительность аудио" value={dashboard.technical_metrics?.audio_duration_seconds ?? '-'} />
          <StatCard label="Общее время обработки" value={dashboard.technical_metrics?.average_processing_time_seconds ?? '-'} />
          <StatCard label="Оценка обработки 1 часа" value={dashboard.technical_metrics?.average_estimated_1h_processing_minutes ?? '-'} />
        </div>
      </details>
    </div>
  );
}
