import type { FormEvent } from 'react';
import { Search, X } from 'lucide-react';
import { useSearchStore } from '../../stores/useSearchStore';
import type { RetrievalMode } from '../../types/competition';

const retrievalModes: RetrievalMode[] = ['hybrid', 'semantic', 'lexical', 'object'];

export function SearchWorkspace() {
  const {
    query,
    setQuery,
    scene2Query,
    setScene2Query,
    retrievalMode,
    setRetrievalMode,
    advancedMode,
    executeSearch,
    isLoading,
  } = useSearchStore();

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void executeSearch();
  };

  const retrievalControl = (
    <label className="retrieval-mode-control">
      <span>Retrieval</span>
      <select
        value={retrievalMode}
        onChange={(event) => setRetrievalMode(event.target.value as RetrievalMode)}
        disabled={isLoading}
      >
        {retrievalModes.map((mode) => (
          <option key={mode} value={mode}>
            {mode}
          </option>
        ))}
      </select>
    </label>
  );

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
            onChange={(event) => setQuery(event.target.value)}
            disabled={isLoading}
          />
          <button type="submit" className="search-submit-btn" disabled={isLoading || !query.trim()}>
            <Search size={18} />
          </button>
        </div>

        <div className="search-options-row">
          <label className="translate-option">
            <input type="checkbox" defaultChecked />
            <span>Translate</span>
          </label>
          <select defaultValue="Vietnamese" className="lang-dropdown">
            <option value="Vietnamese">Vietnamese</option>
            <option value="English">English</option>
          </select>
          {retrievalControl}
        </div>
      </form>
    );
  }

  return (
    <form className="dual-vision-search-header" onSubmit={handleSubmit}>
      <div className="dual-search-col">
        <div className="curved-search-input red-border">
          <span className="hourglass-tag">1</span>
          <input
            type="text"
            placeholder="Describe the scene you are looking for..."
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={isLoading}
          />
          {query && (
            <button type="button" className="btn-clear-x" onClick={() => setQuery('')}>
              <X size={14} />
            </button>
          )}
        </div>
        <div className="dual-sub-options">
          <label><input type="checkbox" defaultChecked /> Translate</label>
          {retrievalControl}
        </div>
      </div>

      <div className="dual-search-col">
        <div className="curved-search-input green-border">
          <span className="hourglass-tag">2</span>
          <input
            type="text"
            placeholder="Describe the next scene (optional)..."
            value={scene2Query}
            onChange={(event) => setScene2Query(event.target.value)}
            disabled={isLoading}
          />
          {scene2Query && (
            <button type="button" className="btn-clear-x" onClick={() => setScene2Query('')}>
              <X size={14} />
            </button>
          )}
        </div>
        <div className="dual-sub-options">
          <span>Temporal scene 2</span>
        </div>
      </div>
    </form>
  );
}
