import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Spinner from '../components/Spinner';
import api from '../lib/api';

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.m4a'];
const MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024;
const MAX_STATUS_POLL_ATTEMPTS = 60;

function getExtension(name) {
  const dot = name.lastIndexOf('.');
  return dot >= 0 ? name.slice(dot).toLowerCase() : '';
}

function getApiErrorMessage(error) {
  const detail = error.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join('; ');
  }
  if (detail && typeof detail === 'object') return JSON.stringify(detail);
  if (error.message) return error.message;
  return 'неизвестная ошибка';
}

export default function UploadPage() {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [consentAccepted, setConsentAccepted] = useState(false);
  const [projectName, setProjectName] = useState('');
  const [meetingDate, setMeetingDate] = useState('');
  const [participantsText, setParticipantsText] = useState('');

  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('idle');
  const [uploadedMeetingId, setUploadedMeetingId] = useState(null);
  const [uploadedFilename, setUploadedFilename] = useState('');
  const [uploadError, setUploadError] = useState('');
  const [analyzeStatus, setAnalyzeStatus] = useState('idle');
  const [analyzeError, setAnalyzeError] = useState('');
  const [queuedTaskId, setQueuedTaskId] = useState('');

  const isUploading = uploadStatus === 'uploading';
  const isAnalyzing = analyzeStatus === 'analyzing';
  const isBusy = isUploading || isAnalyzing;

  const validateBeforeUpload = (file) => {
    if (!consentAccepted) return 'Подтвердите согласие на обработку данных.';
    if (!projectName.trim()) return 'Укажите проект или направление.';
    if (!meetingDate.trim()) return 'Укажите дату встречи.';
    if (!file) return 'Выберите аудиофайл.';
    if (!ALLOWED_EXTENSIONS.includes(getExtension(file.name))) return 'Поддерживаются только MP3, WAV и M4A.';
    if (file.size <= 0) return 'Пустой аудиофайл не может быть загружен.';
    if (file.size > MAX_FILE_SIZE_BYTES) return 'Файл превышает максимальный размер 2 ГБ.';
    return '';
  };

  const buildMeetingTitle = () => {
    const project = projectName.trim();
    return meetingDate ? `${project} — ${meetingDate}` : project;
  };

  const uploadSelectedFile = async (file) => {
    setUploadError('');
    setAnalyzeError('');

    const validationError = validateBeforeUpload(file);
    if (validationError) {
      setUploadStatus('error');
      setUploadError(validationError);
      return;
    }

    try {
      setUploadStatus('uploading');
      setUploadedMeetingId(null);
      setUploadedFilename('');

      const formData = new FormData();
      formData.append('file', file);
      formData.append('meeting_title', buildMeetingTitle());
      formData.append('project_name', projectName.trim());
      formData.append('meeting_date', meetingDate || '');
      formData.append('participants', participantsText.trim());

      const response = await api.post('/upload', formData);
      const payload = response.data || {};
      const meetingId = payload.meeting_id || payload.id || payload.record_id;
      if (!meetingId) {
        throw new Error('backend did not return meeting_id');
      }

      setUploadedMeetingId(meetingId);
      setUploadedFilename(payload.filename || payload.original_filename || file.name);
      setUploadStatus('uploaded');
      if (payload.status === 'queued') {
        setQueuedTaskId(payload.task_id || '');
        setAnalyzeStatus('analyzing');
      } else {
        setQueuedTaskId('');
        setAnalyzeStatus('idle');
      }
    } catch (error) {
      setUploadStatus('error');
      setUploadError(`Не удалось загрузить файл: ${getApiErrorMessage(error)}`);
    }
  };

  const handleFileChange = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    await uploadSelectedFile(file);
    event.target.value = '';
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    await uploadSelectedFile(file);
  };

  const openFilePicker = () => {
    const validationError = validateBeforeUpload({ name: 'placeholder.mp3', size: 1 });
    if (validationError && validationError !== 'Выберите аудиофайл.') {
      setUploadStatus('error');
      setUploadError(validationError);
      return;
    }
    setUploadError('');
    fileInputRef.current?.click();
  };

  const startAnalyze = async () => {
    if (!uploadedMeetingId) return;

    try {
      setAnalyzeError('');
      setAnalyzeStatus('analyzing');
      await api.post(`/meetings/${uploadedMeetingId}/analyze`, null, { timeout: 3600000 });
      setAnalyzeStatus('completed');
    } catch (error) {
      setAnalyzeStatus('error');
      setAnalyzeError(`Не удалось запустить анализ: ${getApiErrorMessage(error)}`);
    }
  };

  useEffect(() => {
    if (!uploadedMeetingId || analyzeStatus !== 'analyzing') return undefined;
    let attempts = 0;
    const timer = window.setInterval(async () => {
      attempts += 1;
      try {
        const response = await api.get(`/meetings/${uploadedMeetingId}/status`);
        const status = response.data?.status;
        const jobStatus = response.data?.job_status;
        if (status === 'completed' || jobStatus === 'success') {
          setAnalyzeStatus('completed');
          setAnalyzeError('');
          return;
        }
        if (status === 'failed' || jobStatus === 'failure') {
          setAnalyzeStatus('error');
          setAnalyzeError('Анализ завершился ошибкой. Проверьте файл или повторите запуск позже.');
          return;
        }
      } catch {
        // ignore transient errors, keep polling
      }
      if (attempts >= MAX_STATUS_POLL_ATTEMPTS) {
        setAnalyzeStatus('error');
        setAnalyzeError('Не удалось получить статус анализа за отведённое время. Попробуйте обновить страницу позже.');
      }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [uploadedMeetingId, analyzeStatus]);

  return (
    <div className="max-w-2xl mx-auto mt-8">
      <h2 className="text-2xl font-bold text-white mb-6">Загрузка аудиозаписи встречи</h2>

      <div className="bg-gray-900 rounded-xl p-8 border border-gray-800 space-y-6">
        {!consentAccepted && (
          <div className="bg-gray-950 border border-gray-700 rounded-xl p-5 space-y-4">
            <h3 className="text-lg font-semibold text-white">Подтверждение правомерности обработки</h3>
            <p className="text-sm text-gray-300">
              Перед загрузкой подтвердите, что у вас есть право на обработку аудиофайла. PM Insights не записывает
              встречи, не использует микрофон и обрабатывает только файл, который вы самостоятельно загружаете.
            </p>
            <label className="flex items-start gap-3 text-sm text-gray-200">
              <input
                type="checkbox"
                checked={consentAccepted}
                onChange={(event) => setConsentAccepted(event.target.checked)}
                className="mt-1"
              />
              <span>Подтверждаю правомерность загрузки и обработки выбранного аудиофайла.</span>
            </label>
          </div>
        )}

        {consentAccepted && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <label className="space-y-1 text-sm text-gray-300">
                <span>Проект / направление</span>
                <input
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                  placeholder="Например: PM Insights"
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white"
                  disabled={isBusy}
                />
              </label>

              <label className="space-y-1 text-sm text-gray-300">
                <span>Дата встречи</span>
                <input
                  type="date"
                  value={meetingDate}
                  onChange={(event) => setMeetingDate(event.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white"
                  disabled={isBusy}
                />
              </label>
            </div>

            <label className="block space-y-1 text-sm text-gray-300">
              <span>Участники и роли</span>
              <textarea
                value={participantsText}
                onChange={(event) => setParticipantsText(event.target.value)}
                placeholder={'Например:\nИван Иванов — руководитель проекта\nАнна Смирнова — аналитик'}
                rows={4}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white"
                disabled={isBusy}
              />
            </label>

            {(uploadError || analyzeError) && (
              <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 text-red-300 text-sm">
                {uploadError || analyzeError}
              </div>
            )}

            <div
              onDragOver={(event) => event.preventDefault()}
              onDrop={isBusy ? undefined : handleDrop}
              onClick={isBusy ? undefined : openFilePicker}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
                isBusy ? 'border-gray-800 cursor-not-allowed opacity-70' : 'border-gray-700 cursor-pointer hover:border-indigo-500'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp3,.wav,.m4a,audio/mpeg,audio/wav,audio/x-m4a"
                className="hidden"
                disabled={isBusy}
                onChange={handleFileChange}
              />

              {selectedFile && analyzeStatus !== 'completed' && (
                <div className="text-sm mb-3">
                  <p className="text-white font-medium">{selectedFile.name}</p>
                  <p className="text-gray-400 mt-1">{(selectedFile.size / 1024 / 1024).toFixed(2)} МБ</p>
                </div>
              )}

              {uploadStatus === 'uploading' && <Spinner text="Файл загружается..." />}
              {analyzeStatus === 'analyzing' && <Spinner text={queuedTaskId ? 'Файл принят в обработку...' : 'Идёт анализ...'} />}

              {!isBusy && uploadStatus !== 'uploaded' && analyzeStatus !== 'completed' && !selectedFile && (
                <div>
                  <p className="text-gray-300">Выберите готовый аудиофайл или перетащите его сюда</p>
                  <p className="text-gray-500 text-xs mt-2">MP3, WAV, M4A</p>
                </div>
              )}

              {!isBusy && uploadStatus === 'uploaded' && selectedFile && (
                <p className="text-emerald-400 text-sm">Файл загружен: {uploadedFilename || selectedFile.name}</p>
              )}

              {!isBusy && analyzeStatus === 'completed' && (
                <div>
                  <p className="text-white font-medium">Анализ завершён</p>
                  {uploadedFilename && <p className="text-gray-400 text-sm mt-1">{uploadedFilename}</p>}
                </div>
              )}
            </div>

            {uploadStatus === 'uploaded' && analyzeStatus === 'idle' && uploadedMeetingId && (
              <button
                type="button"
                onClick={startAnalyze}
                disabled={isAnalyzing}
                className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white font-medium py-3 rounded-lg"
              >
                Запустить анализ
              </button>
            )}

            {analyzeStatus === 'completed' && uploadedMeetingId && (
              <button
                type="button"
                onClick={() => navigate(`/result/${uploadedMeetingId}`)}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium py-3 rounded-lg"
              >
                Открыть результат
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}
