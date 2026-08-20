import { useSearchStore } from '../../stores/useSearchStore';
import { KeyframeCard } from './KeyframeCard';

export function ResultGrid() {
  const { results, advancedMode, runMockSearch, status, error } = useSearchStore();
  const mockMode = import.meta.env.VITE_USE_MOCK_SEARCH === 'true';

  if (!advancedMode) {
    return (
      <div className="okchok-grid-wrapper">
        <div className="okchok-grid-topbar">
          <div className="grid-heading-text">
            <span>•</span> Ranked Keyframes
          </div>
          {mockMode && (
            <button type="button" className="btn-fixture-search" onClick={runMockSearch}>
              Fixture Search (dev)
            </button>
          )}
        </div>

        <div className="okchok-cards-layout">
          {status === 'loading' && <p className="grid-state-message">Searching…</p>}
          {status === 'error' && <p className="grid-state-message error">{error}</p>}
          {status === 'success-empty' && <p className="grid-state-message">No matching keyframes.</p>}
          {status === 'idle' && <p className="grid-state-message">Enter a query to search.</p>}
          {results.map((item, index) => (
            <KeyframeCard key={item.keyframe_id} item={item} index={index} />
          ))}
        </div>
      </div>
    );
  }

  const groupedVideos = results.reduce<Record<string, typeof results>>((groups, item) => {
    if (!groups[item.video_id]) groups[item.video_id] = [];
    groups[item.video_id].push(item);
    return groups;
  }, {});

  return (
    <div className="vision-grouped-results">
      {status === 'loading' && <p className="grid-state-message">Searching…</p>}
      {status === 'error' && <p className="grid-state-message error">{error}</p>}
      {status === 'success-empty' && <p className="grid-state-message">No matching keyframes.</p>}
      {status === 'idle' && <p className="grid-state-message">Enter a query to search.</p>}
      {Object.entries(groupedVideos).map(([videoId, frames]) => (
        <div key={videoId} className="vision-video-row-lane">
          <div className="vision-row-tag">{videoId}</div>
          <div className="vision-cards-horizontal">
            {frames.map((frame, index) => (
              <KeyframeCard key={frame.keyframe_id} item={frame} index={index} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
