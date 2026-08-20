import { useEffect, useRef } from 'react';
import type { KeyboardEvent } from 'react';
import { Search, Languages } from 'lucide-react';
import type { ModelEngine } from '../../types/competition';

interface QueryBarProps {
  id: string;
  variant?: 'scene1' | 'scene2';
  placeholder?: string;
  query: string;
  onQueryChange: (val: string) => void;
  onSearch: () => void;
  isLoading?: boolean;
  modelEngine: ModelEngine;
  onModelEngineChange: (engine: ModelEngine) => void;
  translateEnabled: boolean;
  onTranslateToggle: (enabled: boolean) => void;
  translateLang: string;
  onTranslateLangChange: (lang: string) => void;
  isMain?: boolean;
}

export function QueryBar({
  id,
  variant = 'scene1',
  placeholder = 'Describe the scene you are looking for...',
  query,
  onQueryChange,
  onSearch,
  isLoading = false,
  modelEngine,
  onModelEngineChange,
  translateEnabled,
  onTranslateToggle,
  translateLang,
  onTranslateLangChange,
  isMain = false,
}: QueryBarProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  // Phím tắt Ctrl+K / Cmd+K focus vào ô search chính
  useEffect(() => {
    if (!isMain) return;
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleGlobalKeyDown);
    return () => window.removeEventListener('keydown', handleGlobalKeyDown);
  }, [isMain]);

  // Bắt phím Enter
  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && query.trim() !== '') {
      e.preventDefault();
      onSearch();
    }
  };

  const borderColor = variant === 'scene2' ? 'border-emerald-500/80' : 'border-rose-500/80';

  return (
    <div className="flex flex-col gap-2 w-full">
      <div className="flex items-center gap-2">
        <div className={`relative flex-1 rounded-full border ${borderColor} bg-white dark:bg-slate-900 shadow-sm flex items-center px-4 py-2`}>
          <Search size={18} className="text-slate-400 mr-2 shrink-0" />
          <input
            ref={inputRef}
            id={id}
            type="text"
            className="w-full bg-transparent text-slate-800 dark:text-slate-100 placeholder-slate-400 focus:outline-none text-sm"
            placeholder={placeholder}
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
        </div>

        {isMain && (
          <button
            type="button"
            onClick={onSearch}
            disabled={isLoading || !query.trim()}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white p-2.5 rounded-lg transition shrink-0"
            title="Search (Enter)"
            aria-label="Search"
          >
            <Search size={18} />
          </button>
        )}
      </div>

      <div className="flex items-center gap-4 text-xs text-slate-600 dark:text-slate-400 px-3">
        <label className="flex items-center gap-1.5 cursor-pointer">
          <input
            type="checkbox"
            checked={translateEnabled}
            onChange={(e) => onTranslateToggle(e.target.checked)}
            className="rounded border-slate-300 text-rose-500"
          />
          <Languages size={14} />
          <select
            value={translateLang}
            onChange={(e) => onTranslateLangChange(e.target.value)}
            className="bg-transparent border border-slate-300 dark:border-slate-700 rounded px-1.5 py-0.5 text-xs"
          >
            <option value="vi">Vietnamese</option>
            <option value="en">English</option>
          </select>
        </label>

        <div className="flex items-center gap-3">
          {(['ClipVideo', 'ClipLAION', 'Aladin', 'use all'] as const).map((engine) => (
            <label key={engine} className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name={`${id}-engine`}
                checked={modelEngine === engine}
                onChange={() => onModelEngineChange(engine)}
              />
              <span>{engine}</span>
            </label>
          ))}
        </div>
      </div>
    </div>
  );
}