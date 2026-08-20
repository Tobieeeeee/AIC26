import type { MouseEvent } from 'react';
import type { KeyframeResult } from '../../types/competition';
import { useSearchStore } from '../../stores/useSearchStore';
import { useSubmissionStore } from '../../stores/useSubmissionStore';

interface KeyframeCardProps {
  item: KeyframeResult;
  index?: number;
}

export function KeyframeCard({ item, index = 0 }: KeyframeCardProps) {
  const { advancedMode, selectedResultId, selectResult, setVideoModalOpen } = useSearchStore();
  const { addItem, items } = useSubmissionStore();

  const isSelected = selectedResultId === item.id;
  const isAdded = items.some((s) => s.id === item.id);

  const handleCardClick = () => {
    selectResult(item.id);
  };

  const handleOpenVideo = (e: MouseEvent) => {
    e.stopPropagation();
    selectResult(item.id);
    setVideoModalOpen(true);
  };

  const handleAdd = (e: MouseEvent) => {
    e.stopPropagation();
    addItem(item);
  };

  /* ------------------------------------------------------------- */
  /* MODE 1: Card khi ADVANCED OFF (To rõ, có ảnh & title đầy đủ)  */
  /* ------------------------------------------------------------- */
  if (!advancedMode) {
    return (
      <div
        className={`okchok-card ${isSelected ? 'selected' : ''}`}
        onClick={handleCardClick}
      >
        <div className="okchok-card-media" onClick={handleOpenVideo} title="Click to open video">
          <span className="card-badge-num">#{index + 1}</span>
          {item.thumbnail_url ? (
            <img src={item.thumbnail_url} alt={item.video_id} />
          ) : (
            <div className="tv-placeholder-box">
              <span style={{ fontSize: '32px' }}></span>
            </div>
          )}
        </div>

        <div className="okchok-card-body">
          <strong className="card-video-id">{item.video_id}</strong>
          <span className="card-frame-id">Frame {item.frame_id}</span>
          <span className="card-similarity">{Math.round((item.score ?? 0.9) * 100)}% similarity</span>
        </div>

        <button
          type="button"
          className={`okchok-add-btn ${isAdded ? 'added' : ''}`}
          onClick={handleAdd}
        >
          {isAdded ? '✓ Added' : '+ Add'}
        </button>
      </div>
    );
  }

  /* ------------------------------------------------------------- */
  /* MODE 2: Card chuẩn VISION-E khi ADVANCED ON                  */
  /* ------------------------------------------------------------- */
  return (
    <div
      className={`vision-grid-card ${isSelected ? 'selected' : ''}`}
      onClick={handleCardClick}
    >
      <div className="vision-card-thumb" onClick={handleOpenVideo} title="Click to open video">
        <img
          src={item.thumbnail_url || 'https://images.unsplash.com/photo-1558981403-c5f9899a28bc?w=400&auto=format&fit=crop&q=60'}
          alt={`${item.video_id}-${item.frame_id}`}
        />
      </div>

      <div className="vision-card-actions">
        <span className="action-frame-num">{String(item.frame_id).padStart(3, '0')}</span>
        <button type="button" onClick={() => alert(`Nearby frames quanh Frame ${item.frame_id}`)} title="View Nearby">⊞</button>
        <button type="button" onClick={handleOpenVideo} title="Play Video" className="btn-play-blue">▶</button>
        <button type="button" onClick={handleOpenVideo} title="Similarity">∿</button>
        <button type="button" onClick={() => alert(`Query objects: ${item.objects?.join(', ')}`)} title="Action">🗲</button>
        <button type="button" onClick={handleOpenVideo} title="Video Browser">🎞</button>
        <button
          type="button"
          className={`btn-add-arrow ${isAdded ? 'added' : ''}`}
          onClick={handleAdd}
          title="Submit Keyframe"
        >
          ⬆
        </button>
      </div>
    </div>
  );
}