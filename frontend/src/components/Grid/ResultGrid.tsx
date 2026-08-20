import { useSearchStore } from '../../stores/useSearchStore';
import { KeyframeCard } from './KeyframeCard';

export function ResultGrid() {
  const { results, advancedMode, runMockSearch } = useSearchStore();

  /* Khi ADVANCED OFF: Lưới phẳng đa cột responsive */
  if (!advancedMode) {
    return (
      <div className="okchok-grid-wrapper">
        <div className="okchok-grid-topbar">
          <div className="grid-heading-text">
            <span>▦</span> Ranked Keyframes
          </div>
          <button type="button" className="btn-fixture-search" onClick={runMockSearch}>
            🔍 Fixture Search
          </button>
        </div>

        <div className="okchok-cards-layout">
          {results.map((item, idx) => (
            <KeyframeCard key={item.id} item={item} index={idx} />
          ))}
        </div>
      </div>
    );
  }

  /* Khi ADVANCED ON: Nhóm kết quả theo Video ID */
  const groupedVideos = results.reduce<Record<string, typeof results>>((acc, item) => {
    if (!acc[item.video_id]) acc[item.video_id] = [];
    acc[item.video_id].push(item);
    return acc;
  }, {});

  return (
    <div className="vision-grouped-results">
      {Object.entries(groupedVideos).map(([videoId, frames]) => (
        <div key={videoId} className="vision-video-row-lane">
          <div className="vision-row-tag">{videoId}</div>
          <div className="vision-cards-horizontal">
            {frames.map((frame, i) => (
              <KeyframeCard key={frame.id} item={frame} index={i} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}