import type { FormEvent } from 'react';
import { Search, X } from 'lucide-react';
import { useSearchStore } from '../../stores/useSearchStore';
import type { ModelEngine } from '../../types/competition';

const modelEngines: ModelEngine[] = ['ClipVideo', 'ClipLAION', 'Aladin', 'use all'];

export function SearchWorkspace() {
  const {
    query,
    setQuery,
    scene2Query,
    setScene2Query,
    modelEngine,
    setModelEngine,
    advancedMode,
    executeSearch,
    isLoading,
  } = useSearchStore();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    executeSearch();
  };

  /* ------------------------------------------------------------- */
  /* GIAO DIỆN CHUẨN KHI ADVANCED OFF (Giống hình cuối cùng)       */
  /* ------------------------------------------------------------- */
  if (!advancedMode) {
    return (
      <form className="okchok-search-card" onSubmit={handleSubmit}>
        <div className="search-input-row">
          <Search size={20} className="search-icon-left" />
          <input
            type="text"
            className="search-input-control"
            placeholder="Describe the scene you are looking for..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" className="search-submit-btn">
            <Search size={18} />
          </button>
        </div>

        <div className="search-options-row">
          <label className="translate-option">
            <input type="checkbox" defaultChecked />
            <span>文A Translate</span>
          </label>

          <select defaultValue="Vietnamese" className="lang-dropdown">
            <option value="Vietnamese">Vietnamese</option>
            <option value="English">English</option>
          </select>

          <div className="models-radio-list">
            {modelEngines.map((m) => (
              <label key={m} className="model-radio-item">
                <input
                  type="radio"
                  name="model-select"
                  value={m}
                  checked={modelEngine === m}
                  onChange={() => setModelEngine(m)}
                />
                <span>{m}</span>
              </label>
            ))}
          </div>
        </div>
      </form>
    );
  }

  /* ------------------------------------------------------------- */
  /* GIAO DIỆN KHI ADVANCED ON (Thanh tìm kiếm đôi)                */
  /* ------------------------------------------------------------- */
  return (
    <form className="dual-vision-search-header" onSubmit={handleSubmit}>
      {/* Scene 1 (Viền Đỏ ⏳) */}
      <div className="dual-search-col">
        <div className="curved-search-input red-border">
          <span className="hourglass-tag">⏳</span>
          <input
            type="text"
            placeholder="Describe the scene you are looking for..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {query && (
            <button type="button" className="btn-clear-x" onClick={() => setQuery('')}>
              <X size={14} />
            </button>
          )}
        </div>
        <div className="dual-sub-options">
          <label><input type="checkbox" defaultChecked /> <span>🌐</span></label>
          <select defaultValue="Italian" className="mini-lang-select">
            <option value="Italian">Italian</option>
            <option value="English">English</option>
            <option value="Vietnamese">Vietnamese</option>
          </select>
          <div className="mini-radios">
            {modelEngines.map((m) => (
              <label key={m}>
                <input
                  type="radio"
                  name="v-model-1"
                  value={m}
                  checked={modelEngine === m}
                  onChange={() => setModelEngine(m)}
                />
                {m}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Scene 2 (Viền Xanh Lục ⌛) */}
      <div className="dual-search-col">
        <div className="curved-search-input green-border">
          <span className="hourglass-tag">⌛</span>
          <input
            type="text"
            placeholder="Describe the next scene (optional)..."
            value={scene2Query}
            onChange={(e) => setScene2Query(e.target.value)}
          />
          {scene2Query && (
            <button type="button" className="btn-clear-x" onClick={() => setScene2Query('')}>
              <X size={14} />
            </button>
          )}
        </div>
        <div className="dual-sub-options">
          <label><input type="checkbox" defaultChecked /> <span>🌐</span></label>
          <select defaultValue="Italian" className="mini-lang-select">
            <option value="Italian">Italian</option>
            <option value="English">English</option>
            <option value="Vietnamese">Vietnamese</option>
          </select>
          <div className="mini-radios">
            {modelEngines.map((m) => (
              <label key={`v2-${m}`}>
                <input type="radio" name="v-model-2" defaultChecked={m === 'use all'} />
                {m}
              </label>
            ))}
          </div>
        </div>
      </div>
    </form>
  );
}