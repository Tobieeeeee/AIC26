import { useEffect } from 'react';
import { X, ChevronLeft, ChevronRight, Plus, Check } from 'lucide-react';
import { useSearchStore } from '../../stores/useSearchStore';
import { useSubmissionStore } from '../../stores/useSubmissionStore';

export function VideoPlayerModal() {
  const { isVideoModalOpen, setVideoModalOpen, results, selectedResultId } = useSearchStore();
  const { addItem, items } = useSubmissionStore();

  const currentItem = results.find((r) => r.keyframe_id === selectedResultId);
  const isAdded = currentItem ? items.some((s) => s.keyframe_id === currentItem.keyframe_id) : false;

  // Đóng modal khi bấm phím ESC
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isVideoModalOpen) {
        setVideoModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isVideoModalOpen, setVideoModalOpen]);

  if (!isVideoModalOpen || !currentItem) return null;

  return (
    <div className="modal-overlay-backdrop" onClick={() => setVideoModalOpen(false)}>
      <div className="modal-dialog-box" onClick={(e) => e.stopPropagation()}>
        {/* Header Modal */}
        <div className="modal-dialog-header">
          <div className="modal-header-title-info">
            <strong>{currentItem.video_id}</strong>
            <span>• Frame {currentItem.frame_id}</span>
            <span>• Timestamp: {currentItem.timestamp}</span>
          </div>
          <button
            type="button"
            className="modal-x-close-btn"
            onClick={() => setVideoModalOpen(false)}
          >
            <X size={22} />
          </button>
        </div>

        {/* Video Player Area */}
        <div className="modal-video-screen-wrap">
          <video
            controls
            autoPlay
            src={currentItem.video_url || 'https://www.w3schools.com/html/mov_bbb.mp4'}
            className="modal-html5-player"
          />
        </div>

        {/* Footer Controls */}
        <div className="modal-dialog-footer">
          <div className="modal-footer-meta">
            <div><strong>Objects:</strong> {currentItem.objects?.join(', ') || 'person, car, tree'}</div>
            <div><strong>Similarity:</strong> {Math.round((currentItem.score || 0.95) * 100)}%</div>
          </div>

          <div className="modal-footer-actions">
            <button
              type="button"
              className="modal-step-btn"
              onClick={() => alert(`Tua lùi frame quanh frame ${currentItem.frame_id - 5}`)}
            >
              <ChevronLeft size={16} /> -5 frames
            </button>
            <button
              type="button"
              className="modal-step-btn"
              onClick={() => alert(`Tua tiến frame quanh frame ${currentItem.frame_id + 5}`)}
            >
              +5 frames <ChevronRight size={16} />
            </button>
            <button
              type="button"
              className={`modal-add-tray-btn ${isAdded ? 'added' : ''}`}
              onClick={() => addItem(currentItem)}
            >
              {isAdded ? <Check size={16} /> : <Plus size={16} />}
              {isAdded ? 'Added to Tray' : 'Add to Submission Tray'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
