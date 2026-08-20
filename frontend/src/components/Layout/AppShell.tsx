import { SearchWorkspace } from '../Search/SearchWorkspace';
import { SpatialPanel } from '../Search/SpatialPanel';
import { ResultGrid } from '../Grid/ResultGrid';
import { VideoPlayerModal } from '../Preview/VideoPlayerModal';
import { SubmissionTray } from '../Submission/SubmissionTray';
import { useSearchStore } from '../../stores/useSearchStore';

export function AppShell() {
  const {
    taskMode,
    setTaskMode,
    simReorder,
    setSimReorder,
    advancedMode,
    setAdvancedMode,
    results,
    selectedResultId,
    setVideoModalOpen,
  } = useSearchStore();

  const selectedItem = results.find((r) => r.id === selectedResultId) || results[0];

  return (
    <div className="app-root-shell">
      {/* 1. Header Top */}
      <header className="main-navbar">
        <div className="navbar-left-brand">
        <div className="navbar-left-brand">
          {/* Thay đổi phần hiển thị badge cũ thành thẻ img chứa logo */}
            <img 
             src="/okchokchok.png" 
           alt="OKCHOK Logo" 
          style={{ height: '60px', objectFit: 'contain' }} 
           />
         <div>
  </div>
</div>
          <div>
            <div className="brand-main-title">AIC26 OKCHOK </div>
            <div className="brand-sub-title">Video retrieval operator console</div>
          </div>
        </div>

        {advancedMode && (
          <div className="navbar-center-search">
            <SearchWorkspace />
          </div>
        )}

        <div className="navbar-right-controls">
          <div className="task-mode-radios">
            {(['KIS', 'AVS / TRAKE', 'Q&A'] as const).map((mode) => (
              <label key={mode} className="radio-label-item">
                <input
                  type="radio"
                  name="app-task-mode"
                  checked={taskMode === (mode === 'AVS / TRAKE' ? 'TRAKE' : mode)}
                  onChange={() => setTaskMode(mode === 'AVS / TRAKE' ? 'TRAKE' : (mode as any))}
                />
                <span>{mode}</span>
              </label>
            ))}
          </div>

          <label className="checkbox-label-item">
            <input
              type="checkbox"
              checked={simReorder}
              onChange={(e) => setSimReorder(e.target.checked)}
            />
            <span>Sim Reorder</span>
          </label>

          <label className="adv-switch-control">
            <span>ADVANCED</span>
            <input
              type="checkbox"
              checked={advancedMode}
              onChange={(e) => setAdvancedMode(e.target.checked)}
            />
            <span className={`adv-status-tag ${advancedMode ? 'active' : ''}`}>
              {advancedMode ? 'ON' : 'OFF'}
            </span>
          </label>
        </div>
      </header>

      {/* 2. Main Workspace */}
      {!advancedMode ? (
        <main className="okchok-two-col-layout">
          <div className="okchok-left-workspace">
            <SearchWorkspace />
            <ResultGrid />
          </div>

          <aside className="okchok-right-sidebar">
            <div className="preview-container-box">
              <div className="preview-heading">Preview</div>
              <div className="preview-body-content">
                {/* Click vào ô preview mở modal phát video */}
                <div
                  className="preview-screen-view"
                  onClick={() => setVideoModalOpen(true)}
                  title="Click to open video player"
                  style={{ cursor: 'pointer' }}
                >
                  <div className="preview-placeholder-tv">
                    <span style={{ fontSize: '32px' }}>📺</span>
                    <p style={{ color: '#2563eb', fontWeight: 600, marginTop: 6 }}>
                      ▶ Click to open video player
                    </p>
                  </div>
                </div>
                <div className="preview-metadata-list">
                  <div><span>video_id</span> <strong>{selectedItem?.video_id || 'L01_V001'}</strong></div>
                  <div><span>frame_id</span> <strong>{selectedItem?.frame_id || 1240}</strong></div>
                  <div><span>timestamp</span> <strong>{selectedItem?.timestamp || '00:02:00'}</strong></div>
                  <div><span>objects</span> <strong>{selectedItem?.objects?.join(', ') || 'person, car, tree'}</strong></div>
                </div>
              </div>
            </div>

            <SubmissionTray />
          </aside>
        </main>
      ) : (
        <div className="vision-advanced-workspace">
          <SpatialPanel />
          <main className="vision-main-grid-lane">
            <ResultGrid />
          </main>
        </div>
      )}

      {!advancedMode && (
        <footer className="footer-status-bar">
          <div>🔌 API status: Online (Mock Fixtures)</div>
          <div>⏱ Similarity reorder enabled</div>
          <div>Backend contract: FastAPI (snake_case)</div>
        </footer>
      )}

      <VideoPlayerModal />
    </div>
  );
}