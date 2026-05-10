export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="bg-red-900/30 border border-red-700 rounded-xl p-4 text-center">
      <p className="text-red-400 text-sm">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 text-sm text-red-300 hover:text-white underline cursor-pointer"
        >
          Повторить
        </button>
      )}
    </div>
  );
}
