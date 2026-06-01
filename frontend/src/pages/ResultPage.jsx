/* eslint-disable react-hooks/set-state-in-effect */
/* eslint-disable react-hooks/exhaustive-deps */
import { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import Spinner from '../components/Spinner';
import api from '../lib/api';

const QA_STATUS = {
  answered: 'дан ответ',
  partial: 'частичный ответ',
  not_answered: 'нет ответа / проигнорирован',
};

const DEADLINE_KIND = {
  task_deadline: 'срок задачи',
  answer_deadline: 'срок из ответа',
  meeting_time: 'время встречи',
  mention: 'упоминание срока',
  commercial_deadline: 'коммерческий срок',
};

const MEETING_TYPE_LABELS = {
  commercial_oil_gas: 'нефтегазовая коммерческая встреча',
  oil_gas_commercial: 'нефтегазовая коммерческая встреча',
  commercial_meeting: 'коммерческая встреча',
  technical_research: 'техническая / исследовательская встреча',
  education_consultation: 'учебная консультация',
  project_meeting: 'проектная встреча',
  mixed: 'смешанная встреча',
  general_discussion: 'общее обсуждение',
  unknown: 'тип встречи не определён',
};

function Section({ title, children }) {
  return (
    <section className="bg-gray-900 rounded-xl p-6 border border-gray-800 space-y-4">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      {children}
    </section>
  );
}

function Details({ title, children }) {
  return (
    <details className="bg-gray-950/50 border border-gray-800 rounded-lg p-4">
      <summary className="cursor-pointer text-sm font-medium text-indigo-300">{title}</summary>
      <div className="mt-4">{children}</div>
    </details>
  );
}

function Empty({ text = 'Нет данных' }) {
  return <p className="text-gray-500 text-sm">{text}</p>;
}

function MetricCard({ label, value }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <p className="text-gray-400 text-xs">{label}</p>
      <p className="text-white text-2xl">{value ?? 0}</p>
    </div>
  );
}

function LimitedList({ items, limit, renderItem, moreTitle }) {
  const visible = (items || []).slice(0, limit);
  const hidden = (items || []).slice(limit);
  return (
    <>
      {visible.map(renderItem)}
      {hidden.length > 0 && (
        <Details title={`${moreTitle} (${hidden.length})`}>
          <div className="space-y-3">{hidden.map(renderItem)}</div>
        </Details>
      )}
    </>
  );
}

function compactText(text, limit = 220) {
  if (!text) return '—';
  const clean = String(text).replace(/\s+/g, ' ').trim();
  if (clean.length <= limit) return clean;
  return `${clean.slice(0, limit).trim()}...`;
}

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  const total = Number(value);
  if (total < 60) return `${total.toFixed(1)} сек`;
  return `${Math.floor(total / 60)} мин ${Math.round(total % 60)} сек`;
}

function formatTimecode(item) {
  if (item?.start === undefined || item?.end === undefined) return '—';
  return `${formatSeconds(item.start)} – ${formatSeconds(item.end)}`;
}

const SECTION_ALIASES = {
  questions_answers: ['questions_answers', 'qa'],
  qa: ['qa', 'questions_answers'],
  research_actions: ['research_actions', 'tasks'],
  topics: ['topics', 'aspects_topics'],
};

function reportSection(result, sectionId) {
  const ids = [sectionId, ...(SECTION_ALIASES[sectionId] || [])];
  return (result?.report_sections || []).find((section) => ids.includes(section.id));
}

function getCleanItems(result, sectionId, cleanField, rawField) {
  const section = reportSection(result, sectionId);
  if (section && Array.isArray(section.items)) return section.items;
  if (Array.isArray(result?.[cleanField])) return result[cleanField];
  if (Array.isArray(result?.[rawField])) return result[rawField];
  return [];
}

function hasCleanItems(result, sectionId, cleanField) {
  const section = reportSection(result, sectionId);
  return Boolean(section && Array.isArray(section.items)) || Array.isArray(result?.[cleanField]);
}

function countAnswers(qaItems, hasCleanQA, metrics) {
  if (!hasCleanQA) return metrics.answers_count ?? qaItems.length;
  return qaItems.filter((item) => (
    ['answered', 'partial'].includes(item.status)
    && (item.answer_summary || item.answer_full || item.answer)
  )).length;
}

function countSentiment(items, label) {
  return (items || []).filter((item) => item.sentiment === label).length;
}

function frequencyFromAspectItems(items) {
  const counts = {};
  (items || []).forEach((item) => {
    (item.aspects || []).forEach((aspect) => {
      counts[aspect] = (counts[aspect] || 0) + 1;
    });
  });
  return counts;
}

function meetingType(result) {
  return result?.meeting_type?.label || result?.analysis_summary?.meeting_type || 'unknown';
}

function meetingTypeLabel(result) {
  const type = meetingType(result);
  return result?.meeting_type?.display_name || MEETING_TYPE_LABELS[type] || type;
}

function isTechnical(result) {
  return ['technical_research', 'education_consultation'].includes(meetingType(result));
}

function isCommercial(result) {
  return ['commercial_meeting', 'commercial_oil_gas', 'oil_gas_commercial'].includes(meetingType(result));
}

function cleanTopicName(value) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  const servicePatterns = [
    /проверьте результаты/i,
    /эврист/i,
    /ручн/i,
    /требу(ет|ют) проверки/i,
    /feedback/i,
  ];
  return servicePatterns.some((pattern) => pattern.test(text)) ? '' : text;
}

function getMainTopicNames(summary, topics, aspectCounts) {
  const names = [
    ...(summary.main_topics || []),
    ...(topics || []).map((topic) => topic.topic_name || topic.title || topic.text),
    ...Object.keys(aspectCounts || {}),
  ]
    .map(cleanTopicName)
    .filter(Boolean);
  return Array.from(new Set(names)).slice(0, 8);
}

function buildNeutralSummary(result, topicNames) {
  const top = topicNames.slice(0, 5);
  const topicText = top.length ? top.join(', ') : 'основные вопросы встречи';
  if (isTechnical(result)) {
    return `На встрече обсуждались ${topicText}. Также затрагивались вопросы проверки подходов, уточнения параметров и дальнейших действий по исследованию.`;
  }
  if (isCommercial(result)) {
    return `На встрече обсуждались ${topicText}. Отдельно выделены задачи, вопросы участников, сроки и ключевые аспекты договорённостей.`;
  }
  if (meetingType(result) === 'project_meeting') {
    return `На встрече обсуждались ${topicText}. В результате выделены задачи, вопросы участников и сроки выполнения.`;
  }
  return `На встрече обсуждались ${topicText}. В отчёте собраны выделенные задачи, вопросы, сроки, темы и тональность обсуждения.`;
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

function sentimentClass(label) {
  if (label === 'positive') return 'border-l-4 border-green-700 bg-green-950/10';
  if (label === 'negative') return 'border-l-4 border-red-800 bg-red-950/10';
  return 'border-l-4 border-gray-700 bg-gray-800/50';
}

function segmentSentiment(segment, sentimentItems, index) {
  if (segment.sentiment || segment.sentiment_label) return segment.sentiment || segment.sentiment_label;
  const byFragment = (sentimentItems || []).find((item) => Number(item.source_fragment) === index + 1);
  if (byFragment?.sentiment) return byFragment.sentiment;
  const byText = (sentimentItems || []).find((item) => item.text && segment.text && item.text.includes(segment.text.slice(0, 40)));
  return byText?.sentiment || 'neutral';
}

function taskStatus(task) {
  if (task.review_required) return 'требует проверки';
  if (task.is_repeated || task.repeated || task.status === 'repeated') return 'повторяющаяся';
  return task.status && task.status !== 'new' ? task.status : 'новая';
}

function itemSourceText(item) {
  return (
    item.source_text
    || item.text
    || item.title
    || item.question_full
    || item.question
    || item.question_title
    || item.deadline
    || item.topic_name
    || item.summary
    || ''
  );
}

function FeedbackControls({ actions, onAction }) {
  return (
    <div className="flex flex-wrap gap-2 pt-3 border-t border-gray-800">
      {actions.map((action) => (
        <button
          key={action.label}
          type="button"
          onClick={() => onAction(action)}
          className="px-2.5 py-1 rounded-md bg-gray-950 text-gray-200 text-xs hover:bg-gray-700 border border-gray-700"
        >
          {action.label}
        </button>
      ))}
    </div>
  );
}

function uniqueResponsibleSides(result) {
  const sides = new Set();
  [
    ...(result.clean_tasks || []),
    ...(result.clean_agreements || []),
    ...(result.clean_commercial_terms || []),
    ...(result.clean_commitments || []),
  ].forEach((item) => {
    if (item.responsible_side || item.side) sides.add(item.responsible_side || item.side);
  });
  (result.clean_responsible_sides || result.responsible_side || []).forEach((item) => {
    if (item.side || item.name || item.responsible_side) sides.add(item.side || item.name || item.responsible_side);
  });
  return Array.from(sides);
}

export default function ResultPage() {
  const { id } = useParams();
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [exportError, setExportError] = useState('');
  const [reviewMode, setReviewMode] = useState(false);
  const [feedbackMessage, setFeedbackMessage] = useState('');

  const fetchResult = () => {
    setLoading(true);
    setError('');
    api.get(`/meetings/${id}/result`)
      .then((response) => setResult(response.data))
      .catch((err) => setError(err.response?.data?.detail || 'Результат ещё не готов'))
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchResult(); }, [id]);

  const transcriptText = useMemo(
    () => (result?.transcript || []).map((segment) => segment.text).filter(Boolean).join(' '),
    [result],
  );

  if (loading) return <Spinner />;
  if (error) return <ErrorMessage message={error} onRetry={fetchResult} />;
  if (!result) return null;

  const metrics = result.metrics || {};
  const metadata = result.metadata || {};
  const processingTime = metadata.processing_time || metrics.processing_time || {};
  const meetingInfo = metadata.meeting_info || {};
  const tasks = isTechnical(result)
    ? (getCleanItems(result, 'research_actions', 'clean_research_actions', 'tasks') || getCleanItems(result, 'tasks', 'clean_tasks', 'tasks'))
    : getCleanItems(result, 'tasks', 'clean_tasks', 'tasks');
  const qaItems = getCleanItems(result, 'questions_answers', 'clean_questions_answers', 'questions_answers');
  const hasCleanQA = hasCleanItems(result, 'questions_answers', 'clean_questions_answers');
  const deadlines = getCleanItems(result, 'deadlines', 'clean_deadlines', 'deadlines');
  const responsibles = getCleanItems(result, 'responsibles', 'clean_responsibles', 'responsibles');
  const responsibleSides = uniqueResponsibleSides(result);
  const aspectCounts = metrics.aspect_frequencies || frequencyFromAspectItems(result.aspects);
  const topics = result.topics || [];
  const positiveCount = metrics.positive_fragments_count ?? countSentiment(result.sentiment, 'positive');
  const negativeCount = metrics.negative_fragments_count ?? countSentiment(result.sentiment, 'negative');
  const neutralCount = countSentiment(result.sentiment, 'neutral');
  const transcriptPreviewLimit = result.display_config?.transcript_preview_chars || 1600;
  const transcriptPreview = transcriptText.length > transcriptPreviewLimit
    ? `${transcriptText.slice(0, transcriptPreviewLimit).trim()}...`
    : transcriptText;
  const summary = result.analysis_summary || {};
  const mainTopicNames = getMainTopicNames(summary, topics, aspectCounts);
  const neutralSummary = buildNeutralSummary(result, mainTopicNames);
  const dynamic = result.dynamic_analysis || {};

  const downloadReport = async (format) => {
    setExportError('');
    try {
      const response = await api.get(`/meetings/${id}/export/${format}`, { responseType: 'blob' });
      downloadBlob(response.data, filenameFromDisposition(response.headers['content-disposition'], `pm_insights_${id}.${format}`));
    } catch (err) {
      setExportError(err.response?.data?.detail || 'Не удалось скачать отчёт.');
    }
  };

  const copyTranscript = async () => {
    await navigator.clipboard.writeText(transcriptText);
  };

  const submitFeedback = async (itemType, item, action) => {
    setFeedbackMessage('');
    const sourceText = itemSourceText(item);
    const editedText = action.edit
      ? window.prompt('Введите исправленный текст', action.defaultText || sourceText)
      : '';
    if (action.edit && editedText === null) return;
    try {
      await api.post(`/meetings/${id}/feedback`, {
        item_type: itemType,
        source_text: sourceText,
        predicted_label: action.predictedLabel || itemType,
        corrected_label: action.correctedLabel,
        corrected_text: action.edit ? editedText : (action.correctedText || sourceText),
        metadata: {
          ...action.metadata,
          source_fragment: item.source_fragment,
          status: item.status,
          responsible: item.responsible,
          deadline: item.deadline,
          question: item.question_full || item.question || item.question_title,
          answer: item.answer_full || item.answer || item.answer_summary,
          topic_name: item.topic_name,
          keywords: item.keywords,
        },
      });
      setFeedbackMessage('Исправление сохранено как разметка для будущего безопасного дообучения.');
    } catch (err) {
      setFeedbackMessage(err.response?.data?.detail || 'Не удалось сохранить исправление.');
    }
  };

  return (
    <div className="max-w-6xl mx-auto mt-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">
            {meetingInfo.meeting_title || result.meeting_title || result.source_audio || result.meeting_id}
          </h2>
          <div className="text-gray-400 text-sm mt-2 space-y-1">
            <p>{meetingInfo.project_name || result.project_name || 'PM Insights'} {meetingInfo.meeting_date || result.meeting_date ? `| ${meetingInfo.meeting_date || result.meeting_date}` : ''}</p>
            <p>Модель ASR: {metadata.asr_model || '—'}</p>
            <p>Длительность: {formatSeconds(metadata.duration_seconds || processingTime.audio_duration_seconds)} | Обработка: {formatSeconds(processingTime.total_processing_seconds)}</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={() => downloadReport('pdf')} className="px-3 py-2 rounded-lg bg-gray-950 text-white hover:bg-gray-800">Скачать PDF</button>
          <button type="button" onClick={() => downloadReport('xlsx')} className="px-3 py-2 rounded-lg bg-gray-950 text-white hover:bg-gray-800">Скачать Excel</button>
          <button type="button" onClick={() => downloadReport('docx')} className="px-3 py-2 rounded-lg bg-gray-950 text-white hover:bg-gray-800">Скачать Word</button>
          <button type="button" onClick={() => setReviewMode((value) => !value)} className="px-3 py-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700">
            {reviewMode ? 'Выключить режим проверки' : 'Режим проверки'}
          </button>
          <Link to="/history" className="px-3 py-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700">Архив</Link>
        </div>
      </div>

      {exportError && <div className="bg-red-950/20 border border-red-900/60 rounded-xl p-3 text-sm text-red-200">{exportError}</div>}
      {reviewMode && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 text-sm text-gray-300">
          Исправления будут сохранены как разметка и могут использоваться для будущего дообучения модели. Модель не переобучается автоматически.
        </div>
      )}
      {feedbackMessage && <div className="bg-gray-900 border border-gray-800 rounded-xl p-3 text-sm text-gray-300">{feedbackMessage}</div>}

      <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-7 gap-3">
        <MetricCard label="Задачи" value={tasks.length} />
        <MetricCard label="Вопросы" value={qaItems.length} />
        <MetricCard label="Ответы" value={countAnswers(qaItems, hasCleanQA, metrics)} />
        <MetricCard label="Ответственные" value={responsibles.length} />
        <MetricCard label="Дедлайны" value={deadlines.length} />
        <MetricCard label="Средний тон" value={metrics.average_sentiment ?? 0} />
        <MetricCard label="Негатив" value={negativeCount} />
      </div>

      <Section title="Краткая сводка">
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-gray-500">Тип встречи</p>
            <p className="text-white">{meetingTypeLabel(result)}</p>
          </div>
          <div>
            <p className="text-gray-500">Основные темы</p>
            <p className="text-white">{mainTopicNames.join(', ') || '—'}</p>
          </div>
        </div>
        <div>
          <p className="text-gray-500 text-sm mb-2">Краткое содержание</p>
          <p className="text-gray-300 text-sm leading-relaxed">{neutralSummary}</p>
        </div>
      </Section>

      <Section title="Задачи">
        {tasks.length ? (
          <LimitedList items={tasks} limit={8} moreTitle="Показать все задачи" renderItem={(task, index) => (
            <div key={index} className="bg-gray-800/50 rounded-lg p-4 text-sm">
              <p className="text-white font-medium">{task.title || task.text}</p>
              {task.summary && <p className="text-gray-300 mt-2">{compactText(task.summary, 260)}</p>}
              <div className="mt-3 grid sm:grid-cols-2 gap-2 text-gray-400">
                <p>Фрагмент: <span className="text-gray-200">{task.source_fragment || '—'}</span></p>
                <p>Ответственный: <span className="text-gray-200">{task.responsible || 'не определён'}</span></p>
                <p>Срок: <span className="text-gray-200">{task.deadline || 'не определён'}</span></p>
                <p>Статус: <span className="text-gray-200">{taskStatus(task)}</span></p>
              </div>
              {reviewMode && (
                <FeedbackControls
                  actions={[
                    { label: 'Верно', correctedLabel: 'task', predictedLabel: 'task' },
                    { label: 'Не задача', correctedLabel: 'other', predictedLabel: 'task' },
                    { label: 'Редактировать', correctedLabel: 'task', predictedLabel: 'task', edit: true },
                  ]}
                  onAction={(action) => submitFeedback('task', task, action)}
                />
              )}
            </div>
          )} />
        ) : <Empty text="Выделенные задачи не найдены." />}
      </Section>

      <Section title="Вопросы и ответы">
        {qaItems.length ? (
          <LimitedList items={qaItems} limit={8} moreTitle="Показать все вопросы и ответы" renderItem={(item, index) => (
            <div key={index} className="bg-gray-800/50 rounded-lg p-4 text-sm space-y-2">
              <p className="text-gray-400">Вопрос</p>
              <p className="text-white">{item.question_title || compactText(item.question, 220)}</p>
              <p className="text-gray-400">Ответ</p>
              <p className="text-gray-200">{item.answer_summary || compactText(item.answer, 320)}</p>
              <p className="text-xs text-gray-500">Статус: {QA_STATUS[item.status] || item.status || '—'}</p>
              <Details title="Показать полный Q/A фрагмент">
                <div className="space-y-3 text-sm">
                  <p className="text-gray-200">{item.question_full || item.question || '—'}</p>
                  <p className="text-gray-300">{item.answer_full || item.answer || '—'}</p>
                </div>
              </Details>
              {reviewMode && (
                <FeedbackControls
                  actions={[
                    { label: 'Верно', correctedLabel: item.status || 'answered', predictedLabel: item.status || 'answered', metadata: { item_type: 'qa' } },
                    { label: 'Неверно', correctedLabel: 'incorrect', predictedLabel: item.status || 'answered', metadata: { item_type: 'qa' } },
                    { label: 'Редактировать ответ', correctedLabel: item.status || 'answered', predictedLabel: item.status || 'answered', edit: true, defaultText: item.answer_summary || item.answer_full || item.answer || '' },
                    { label: 'Нет ответа', correctedLabel: 'not_answered', predictedLabel: item.status || 'answered', correctedText: item.question_title || item.question || '' },
                  ]}
                  onAction={(action) => submitFeedback('qa', item, action)}
                />
              )}
            </div>
          )} />
        ) : <Empty />}
      </Section>

      <Section title="Ответственные и сроки">
        <div className="grid lg:grid-cols-2 gap-5">
          <div className="space-y-2">
            <h4 className="text-white font-semibold">Ответственные</h4>
            {responsibles.length ? responsibles.slice(0, 10).map((item, index) => (
              <div key={index} className="bg-gray-800/50 rounded-lg p-3 text-sm">
                <p className="text-gray-200">{item.name || item.responsible || (item.responsibles || []).join(', ')}</p>
                {reviewMode && (
                  <FeedbackControls
                    actions={[
                      { label: 'Верно', correctedLabel: 'responsible', predictedLabel: 'responsible' },
                      { label: 'Не ответственный', correctedLabel: 'other', predictedLabel: 'responsible' },
                      { label: 'Редактировать', correctedLabel: 'responsible', predictedLabel: 'responsible', edit: true },
                    ]}
                    onAction={(action) => submitFeedback('responsible', item, action)}
                  />
                )}
              </div>
            )) : <Empty text="Ответственные не определены." />}
            {responsibleSides.length > 0 && (
              <p className="text-sm text-gray-400">Ответственные стороны: {responsibleSides.join(', ')}</p>
            )}
          </div>
          <div className="space-y-2">
            <h4 className="text-white font-semibold">Дедлайны</h4>
            {deadlines.length ? (
              <LimitedList items={deadlines} limit={10} moreTitle="Показать все дедлайны" renderItem={(item, index) => (
                <div key={index} className="bg-gray-800/50 rounded-lg p-3 text-sm">
                  <p className="text-gray-200">Срок: {item.deadline || (item.deadlines || []).join(', ') || '—'}</p>
                  <p className="text-gray-400">Контекст: {item.context || compactText(item.text, 140)}</p>
                  <p className="text-gray-500 text-xs">Тип: {DEADLINE_KIND[item.kind] || item.kind || '—'} | Фрагмент: {item.source_fragment || '—'}</p>
                  {reviewMode && (
                    <FeedbackControls
                      actions={[
                        { label: 'Верно', correctedLabel: 'deadline', predictedLabel: 'deadline' },
                        { label: 'Не срок', correctedLabel: 'other', predictedLabel: 'deadline' },
                        { label: 'Редактировать', correctedLabel: 'deadline', predictedLabel: 'deadline', edit: true },
                      ]}
                      onAction={(action) => submitFeedback('deadline', item, action)}
                    />
                  )}
                </div>
              )} />
            ) : <Empty />}
          </div>
        </div>
      </Section>

      <Section title="Аспекты и темы">
        <div className="grid lg:grid-cols-2 gap-5">
          <div>
            <h4 className="text-white font-semibold mb-3">Аспекты обсуждения</h4>
            {Object.keys(aspectCounts || {}).length ? (
              <div className="space-y-2 text-sm">
                {Object.entries(aspectCounts).sort((a, b) => b[1] - a[1]).slice(0, 10).map(([aspect, count]) => (
                  <div key={aspect} className="flex justify-between gap-3 border-b border-gray-800 pb-2">
                    <span className="text-gray-300">{aspect}</span>
                    <span className="text-gray-400">{count}</span>
                    {reviewMode && (
                      <FeedbackControls
                        actions={[
                          { label: 'Верно', correctedLabel: 'aspect', predictedLabel: 'aspect' },
                          { label: 'Не относится', correctedLabel: 'other', predictedLabel: 'aspect' },
                          { label: 'Редактировать', correctedLabel: 'aspect', predictedLabel: 'aspect', edit: true, defaultText: aspect },
                        ]}
                        onAction={(action) => submitFeedback('aspect', { title: aspect, source_text: aspect }, action)}
                      />
                    )}
                  </div>
                ))}
              </div>
            ) : <Empty />}
          </div>
          <div>
            <h4 className="text-white font-semibold mb-3">Темы</h4>
            {topics.length ? topics.slice(0, 10).map((topic, index) => (
              <div key={index} className="bg-gray-800/50 rounded-lg p-3 text-sm mb-2">
                <p className="text-white">{topic.topic_name}</p>
                <p className="text-gray-400">Ключевые слова: {(topic.keywords || []).join(', ') || '—'}</p>
                <p className="text-gray-500 text-xs">Фрагментов: {(topic.fragments || []).length}</p>
                {reviewMode && (
                  <FeedbackControls
                    actions={[
                      { label: 'Верно', correctedLabel: 'topic', predictedLabel: 'topic' },
                      { label: 'Не относится', correctedLabel: 'other', predictedLabel: 'topic' },
                      { label: 'Редактировать', correctedLabel: 'topic', predictedLabel: 'topic', edit: true, defaultText: topic.topic_name },
                    ]}
                    onAction={(action) => submitFeedback('topic', topic, action)}
                  />
                )}
              </div>
            )) : <Empty />}
          </div>
        </div>
      </Section>

      <Section title="Тональность">
        <div className="grid sm:grid-cols-4 gap-3">
          <MetricCard label="Позитив" value={positiveCount} />
          <MetricCard label="Нейтрально" value={neutralCount} />
          <MetricCard label="Негатив" value={negativeCount} />
          <MetricCard label="Средний тон" value={metrics.average_sentiment ?? 0} />
        </div>
        <Details title="Показать фрагменты с тональностью">
          <div className="space-y-2">
            {(result.sentiment || []).map((item, index) => (
              <div key={index} className={`${sentimentClass(item.sentiment)} rounded-lg p-3 text-sm`}>
                <p className="text-gray-200">{item.text}</p>
                <p className="text-gray-500 text-xs mt-1">{item.sentiment} ({item.score})</p>
                {reviewMode && (
                  <FeedbackControls
                    actions={[
                      { label: 'Позитив', correctedLabel: 'positive', predictedLabel: item.sentiment || 'neutral' },
                      { label: 'Нейтрально', correctedLabel: 'neutral', predictedLabel: item.sentiment || 'neutral' },
                      { label: 'Негатив', correctedLabel: 'negative', predictedLabel: item.sentiment || 'neutral' },
                    ]}
                    onAction={(action) => submitFeedback('sentiment', item, action)}
                  />
                )}
              </div>
            ))}
            {!result.sentiment?.length && <Empty />}
          </div>
        </Details>
      </Section>

      <Section title="Динамика по серии встреч">
        {dynamic.available ? (
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3 text-sm">
            <MetricCard label="Предыдущих встреч" value={dynamic.previous_meetings_count} />
            <MetricCard label="Δ задач" value={dynamic.tasks_delta} />
            <MetricCard label="Δ среднего тона" value={dynamic.average_sentiment_delta} />
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
              <p className="text-gray-400 text-xs">Повторяющиеся аспекты</p>
              <p className="text-white text-sm mt-2">{(dynamic.repeated_topics || []).join(', ') || '—'}</p>
            </div>
          </div>
        ) : <p className="text-gray-500 text-sm">{dynamic.message || 'Для этой серии встреч пока нет истории для динамического анализа.'}</p>}
      </Section>

      <Section title="Служебная информация">
        <Details title="Показать технические метрики">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 text-sm">
            <MetricCard label="Длительность" value={formatSeconds(processingTime.audio_duration_seconds || metadata.duration_seconds)} />
            <MetricCard label="Обработка" value={formatSeconds(processingTime.total_processing_seconds)} />
            <MetricCard label="NLP время" value={formatSeconds(processingTime.nlp_seconds)} />
            <MetricCard label="Модель ASR" value={metadata.asr_model || '—'} />
          </div>
        </Details>
      </Section>

      <Section title="Транскрипт">
        <button type="button" onClick={copyTranscript} className="px-3 py-2 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700">
          Скопировать полный транскрипт
        </button>
        <div className="bg-gray-950/50 border border-gray-800 rounded-lg p-4 text-gray-200 text-sm leading-relaxed whitespace-pre-wrap max-h-96 overflow-y-auto">
          {transcriptText || transcriptPreview || 'Нет данных'}
        </div>
        <Details title="Показать сегменты с таймкодами">
          <div className="space-y-2">
            {(result.transcript || []).map((segment, index) => (
              <div key={index} className={`${sentimentClass(segmentSentiment(segment, result.sentiment, index))} rounded-lg p-3 text-sm`}>
                <span className="text-gray-500">[{formatTimecode(segment)}]</span>{' '}
                <span className="text-gray-200">{segment.text}</span>
              </div>
            ))}
          </div>
        </Details>
      </Section>
    </div>
  );
}
