import { SearchWorkspace } from '../Search/SearchWorkspace';
import { SpatialPanel } from '../Search/SpatialPanel';
import { ResultGrid } from '../Grid/ResultGrid';
import { VideoPlayerModal } from '../Preview/VideoPlayerModal';
import { SubmissionTray } from '../Submission/SubmissionTray';
import { isMockSearchEnabled, useSearchStore } from '../../stores/useSearchStore';

const taskModes = [
  { value: 'kis' as const, label: 'KIS' },
  { value: 'trake' as const, label: 'AVS / TRAKE' },
  { value: 'qa' as const, label: 'Q&A' },
];

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
    status,
  } = useSearchStore();

  const selectedItem = results.find((result) => result.keyframe_id === selectedResultId);

  return (
    <div className="app-root-shell">
      <header className="main-navbar">
        <div className="navbar-left-brand">
          <img src="/okchokchok.png" alt="OKCHOK Logo" style={{ height: '60px', objectFit: 'contain' }} />
          <div>
            <div className="brand-main-title">AIC26 OKCHOK</div>
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
            {taskModes.map((mode) => (
              <label key={mode.value} className="radio-label-item">
                <input
                  type="radio"
                  name="app-task-mode"
                  checked={taskMode === mode.value}
                  onChange={() => setTaskMode(mode.value)}
                />
                <span>{mode.label}</span>
              </label>
            ))}
          </div>

          <label className="checkbox-label-item">
            <input
              type="checkbox"
              checked={simReorder}
              onChange={(event) => setSimReorder(event.target.checked)}
            />
            <span>Sim Reorder</span>
          </label>

          <label className="adv-switch-control">
            <span>ADVANCED</span>
            <input
              type="checkbox"
              checked={advancedMode}
              onChange={(event) => setAdvancedMode(event.target.checked)}
            />
            <span className={`adv-status-tag ${advancedMode ? 'active' : ''}`}>
              {advancedMode ? 'ON' : 'OFF'}
            </span>
          </label>
        </div>
      </header>

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
                <button
                  type="button"
                  className="preview-screen-view"
                  onClick={() => selectedItem && setVideoModalOpen(true)}
                  disabled={!selectedItem}
                  title={selectedItem ? 'Open video player' : 'Select a keyframe first'}
                >
                  <div className="preview-placeholder-tv">
                    <span style={{ fontSize: '32px' }}>TV</span>
                    <p style={{ color: '#2563eb', fontWeight: 600, marginTop: 6 }}>
                      {selectedItem ? 'Open video player' : 'Select a keyframe'}
                    </p>
                  </div>
                </button>
                <div className="preview-metadata-list">
                  <div><span>keyframe_id</span> <strong>{selectedItem?.keyframe_id || '—'}</strong></div>
                  <div><span>video_id</span> <strong>{selectedItem?.video_id || '—'}</strong></div>
                  <div><span>frame_id</span> <strong>{selectedItem?.frame_id ?? '—'}</strong></div>
                  <div><span>timestamp</span> <strong>{selectedItem?.timestamp || '—'}</strong></div>
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
          <div>API status: {status === 'error' ? 'Error' : 'Ready'}</div>
          <div>{isMockSearchEnabled ? 'Fixture mode enabled (development)' : 'Production API mode'}</div>
          <div>Backend contract: unified POST /search (Task06A)</div>
        </footer>
      )}

      <VideoPlayerModal />
    </div>
  );
}
