/* eslint-disable react-hooks/set-state-in-effect */
/* eslint-disable react-hooks/exhaustive-deps */
import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import api from '../lib/api';
import ErrorMessage from '../components/ErrorMessage';

function EmptyState({ children }) {
  return <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 text-sm text-gray-400">{children}</div>;
}

function DashboardSkeleton() {
  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6 animate-pulse">
      <div className="space-y-2">
        <div className="h-8 w-64 bg-gray-800 rounded" />
        <div className="h-4 w-96 bg-gray-800 rounded" />
      </div>
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="h-10 w-full max-w-xl bg-gray-800 rounded-lg" />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {Array.from({ length: 8 }).map((_, index) => (
          <div key={index} className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-2">
            <div className="h-3 w-16 bg-gray-800 rounded" />
            <div className="h-6 w-12 bg-gray-800 rounded" />
          </div>
        ))}
      </div>
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <div className="h-5 w-48 bg-gray-800 rounded" />
        <div className="h-72 bg-gray-800 rounded" />
      </div>
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
        <div className="h-5 w-64 bg-gray-800 rounded" />
        <div className="h-24 bg-gray-800 rounded" />
      </div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p className="text-gray-400 text-xs">{label}</p>
      <p className="text-white text-2xl font-semibold mt-1">{value ?? 0}</p>
    </div>
  );
}

const MIN_TREND_POINTS = 2;

function ChartCard({ title, description, children }) {
  return (
    <section className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-white">{title}</h3>
        <p className="text-sm text-gray-400 mt-1">{description}</p>
      </div>
      {children}
    </section>
  );
}

function buildTrendData(sentimentTrend, taskTrend) {
  return (sentimentTrend || []).map((item, index) => ({
    date: item.date,
    meeting_title: item.meeting_title || 'Встреча',
    average_sentiment: Number(item.average_sentiment || 0),
    tasks_count: Number((taskTrend || [])[index]?.tasks_count || 0),
  }));
}

function TrendChart({ data }) {
  if (data.length < MIN_TREND_POINTS) {
    return <EmptyState>Недостаточно данных для графика — нужно минимум 2 встречи с результатами.</EmptyState>;
  }
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 10, right: 24, bottom: 10, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis dataKey="date" stroke="#9ca3af" tick={{ fill: '#9ca3af', fontSize: 12 }} />
        <YAxis
          yAxisId="sentiment"
          stroke="#818cf8"
          tick={{ fill: '#818cf8', fontSize: 12 }}
          domain={[-1, 1]}
        />
        <YAxis
          yAxisId="tasks"
          orientation="right"
          stroke="#34d399"
          tick={{ fill: '#34d399', fontSize: 12 }}
          allowDecimals={false}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: 8 }}
          labelStyle={{ color: '#e5e7eb' }}
          itemStyle={{ color: '#e5e7eb' }}
          formatter={(value, name) => [value, name]}
          labelFormatter={(label, payload) => `${label} · ${payload?.[0]?.payload?.meeting_title || ''}`}
        />
        <Legend wrapperStyle={{ color: '#9ca3af' }} />
        <Line
          yAxisId="sentiment"
          type="monotone"
          dataKey="average_sentiment"
          name="Средний тон"
          stroke="#818cf8"
          strokeWidth={2}
          dot={{ r: 4 }}
        />
        <Line
          yAxisId="tasks"
          type="monotone"
          dataKey="tasks_count"
          name="Задачи"
          stroke="#34d399"
          strokeWidth={2}
          dot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
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
            className="text-white"
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

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorMessage message={error} onRetry={() => load(project)} />;
  if (!dashboard) return null;

  const handleProjectChange = (event) => {
    const value = event.target.value;
    setProject(value);
    load(value);
  };

  const technicalMetrics = dashboard.technical_metrics || {};
  const meetingGroups = dashboard.meeting_groups || [];
  const trendData = buildTrendData(dashboard.sentiment_trend, dashboard.task_trend);

  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-white">Сводка по проекту</h2>
        <p className="text-gray-400 text-sm mt-2">
          Динамика настроения, количества задач и обсуждаемых аспектов по выбранному проекту.
        </p>
      </div>

      <section className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <label className="text-sm text-gray-300 block max-w-xl">
          <span className="block mb-1 font-medium">Проект</span>
          <select
            value={project}
            onChange={handleProjectChange}
            className="w-full border border-gray-700 rounded-lg px-3 py-2 bg-gray-800 text-white"
          >
            <option value="">Все проекты</option>
            {projects.map((item) => (
              <option key={item.project_key} value={item.project_key}>{item.project_name}</option>
            ))}
          </select>
        </label>
      </section>

      {!projects.length && <EmptyState>Пока нет обработанных встреч для построения сводки.</EmptyState>}

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Встречи" value={summary.meetings_count} />
        <StatCard label="Завершено" value={summary.completed_count} />
        <StatCard label="Задачи" value={summary.tasks_count} />
        <StatCard label="Вопросы" value={summary.questions_count} />
        <StatCard label="Ответы" value={summary.answers_count} />
        <StatCard label="Дедлайны" value={summary.deadlines_count} />
        <StatCard label="Средний тон" value={summary.average_sentiment} />
        <StatCard label="Негатив" value={summary.negative_count ?? '-'} />
      </div>

      <ChartCard title="Динамика встреч" description="Средняя тональность и количество задач по встречам во времени.">
        <TrendChart data={trendData} />
      </ChartCard>

      <ChartCard title="Облако обсуждаемых аспектов" description="Самые обсуждаемые аспекты и темы по выбранному проекту.">
        <WordCloud words={dashboard.aspect_word_cloud || []} />
      </ChartCard>

      <ChartCard title="Технические метрики" description="Производительность обработки встреч за выбранный период.">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="Встречи (тех.)" value={technicalMetrics.meetings_count} />
          <StatCard label="Всего задач" value={technicalMetrics.total_tasks} />
          <StatCard label="Всего вопросов" value={technicalMetrics.total_questions} />
          <StatCard
            label="Среднее время обработки, с"
            value={technicalMetrics.average_processing_time_seconds ?? '-'}
          />
          <StatCard
            label="Оценка на 1 час аудио, мин"
            value={technicalMetrics.average_estimated_1h_processing_minutes ?? '-'}
          />
        </div>
      </ChartCard>

      <ChartCard title="Группы встреч" description="Встречи, объединённые в серии/группы для отслеживания динамики.">
        {meetingGroups.length === 0 ? (
          <EmptyState>Группировка встреч пока не рассчитана.</EmptyState>
        ) : (
          <ul className="space-y-2">
            {meetingGroups.map((group) => (
              <li key={group.group_key || group.name} className="text-sm text-gray-300">
                {group.name || group.group_key} — {group.meetings_count ?? group.meetings?.length ?? 0} встреч
              </li>
            ))}
          </ul>
        )}
      </ChartCard>

    </div>
  );
}
