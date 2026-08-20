import { Trash2, RotateCcw, Search, Upload } from 'lucide-react';
import { useSearchStore } from '../../stores/useSearchStore';

const iconsList = [
  { id: 'person', label: '👤' },
  { id: 'man', label: '👨' },
  { id: 'woman', label: '👩' },
  { id: 'baby', label: '👶' },
  { id: 'music', label: '🎵' },
  { id: 'tennis', label: '🎾' },
  { id: 'glasses', label: '👓' },
  { id: 'bicycle', label: '🚲' },
  { id: 'motorcycle', label: '🏍️' },
  { id: 'car', label: '🚗' },
  { id: 'truck', label: '🚚' },
  { id: 'boat', label: '🚤' },
  { id: 'airplane', label: '✈️' },
  { id: 'bench', label: '🪑' },
  { id: 'chair', label: '💺' },
  { id: 'couch', label: '🛋️' },
  { id: 'umbrella', label: '⛱️' },
  { id: 'cup', label: '🍷' },
  { id: 'coffee', label: '☕' },
  { id: 'phone', label: '📱' },
  { id: 'laptop', label: '💻' },
  { id: 'tv', label: '📺' },
  { id: 'clock', label: '⏰' },
  { id: 'traffic', label: '🚦' },
  { id: 'cat', label: '🐈' },
  { id: 'dog', label: '🐕' },
  { id: 'bird', label: '🐦' },
  { id: 'flower', label: '🌸' },
  { id: 'tree', label: '🌳' },
];

const colorList = [
  { name: 'BLACK', bg: '#000000', text: '#fff' },
  { name: 'BLUE', bg: '#2563eb', text: '#fff' },
  { name: 'BROWN', bg: '#78350f', text: '#fff' },
  { name: 'GREEN', bg: '#16a34a', text: '#fff' },
  { name: 'GREY', bg: '#64748b', text: '#fff' },
  { name: 'ORANGE', bg: '#ea580c', text: '#fff' },
  { name: 'PINK', bg: '#db2777', text: '#fff' },
  { name: 'PURPLE', bg: '#9333ea', text: '#fff' },
  { name: 'RED', bg: '#dc2626', text: '#fff' },
  { name: 'WHITE', bg: '#ffffff', text: '#000' },
  { name: 'YELLOW', bg: '#eab308', text: '#000' },
];

export function SpatialPanel() {
  const {
    scene1Spatial,
    scene2Spatial,
    updateSpatialState,
    toggleObjectInSpatial,
    toggleColorInSpatial,
    clearSpatialState,
    visualQuery,
    updateVisualQuery,
    executeSearch,
  } = useSearchStore();

  const handleUploadImage = () => {
    const url = prompt('Dán đường dẫn ảnh truy vấn (Image URL):', 'https://');
    if (url) {
      updateVisualQuery({ imageUrl: url });
      executeSearch();
    }
  };

  return (
    <aside className="advanced-sidebar-bar">
      {/* 1. Icon Grid */}
      <div className="adv-icon-grid">
        {iconsList.map((obj) => (
          <button
            key={obj.id}
            type="button"
            className={`adv-icon-chip ${scene1Spatial.selectedObjects.includes(obj.id) ? 'active' : ''}`}
            onClick={() => toggleObjectInSpatial('scene1', obj.id)}
            title={obj.id}
          >
            <span>{obj.label}</span>
          </button>
        ))}
      </div>

      {/* 2. Color Bar */}
      <div className="adv-color-strip">
        {colorList.map((c) => (
          <button
            key={c.name}
            type="button"
            className={`adv-color-chip ${scene1Spatial.selectedColors.includes(c.name) ? 'active' : ''}`}
            style={{
              backgroundColor: c.bg,
              color: c.text,
              border: c.name === 'WHITE' ? '1px solid #cbd5e1' : 'none',
            }}
            onClick={() => toggleColorInSpatial('scene1', c.name)}
          >
            {c.name}
          </button>
        ))}
      </div>

      {/* 3. ⏳ Objects & colors of the scene */}
      <div className="adv-spatial-box red-bordered">
        <div className="adv-box-header red-color">
          <span className="adv-box-title">⏳ Objects & colors of the scene</span>
          <div className="adv-box-radios">
            <label><input type="radio" name="s1-en" checked={scene1Spatial.enabled} onChange={() => updateSpatialState('scene1', { enabled: true })} /> Enabled</label>
            <label><input type="radio" name="s1-en" checked={!scene1Spatial.enabled} onChange={() => updateSpatialState('scene1', { enabled: false })} /> Disabled</label>
            <span style={{ margin: '0 4px' }}>Objects in</span>
            <label><input type="radio" name="s1-lg" checked={scene1Spatial.logic === 'AND'} onChange={() => updateSpatialState('scene1', { logic: 'AND' })} /> AND</label>
            <label><input type="radio" name="s1-lg" checked={scene1Spatial.logic === 'OR'} onChange={() => updateSpatialState('scene1', { logic: 'OR' })} /> OR</label>
          </div>
        </div>

        <div className="adv-canvas-viewport">
          <div className="adv-dashed-grid"></div>
          {scene1Spatial.selectedObjects.length > 0 && (
            <div className="canvas-tags-overlay">
              {scene1Spatial.selectedObjects.map((tag) => (
                <span key={tag} className="tag-item" onClick={() => toggleObjectInSpatial('scene1', tag)}>
                  {tag} ✕
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="adv-box-footer">
          <textarea
            className="adv-max-obj-input"
            value={scene1Spatial.maxObjectsText}
            onChange={(e) => updateSpatialState('scene1', { maxObjectsText: e.target.value })}
            placeholder="Max Obj: e.g.: 2 person 3 car 0 dog, means at most 2 persons, 3 cars, no dogs"
          />
          <div className="adv-actions-stack">
            <button type="button" className="adv-btn-tool red" onClick={() => clearSpatialState('scene1')} title="Clear"><Trash2 size={16} /></button>
            <button type="button" className="adv-btn-tool green" onClick={() => clearSpatialState('scene1')} title="Reset"><RotateCcw size={16} /></button>
          </div>
        </div>
      </div>

      {/* 4. ⌛ Objects & color next scene */}
      <div className="adv-spatial-box green-bordered">
        <div className="adv-box-header green-color">
          <span className="adv-box-title">⌛ Objects & color next scene</span>
          <div className="adv-box-radios">
            <label><input type="radio" name="s2-en" checked={scene2Spatial.enabled} onChange={() => updateSpatialState('scene2', { enabled: true })} /> Enabled</label>
            <label><input type="radio" name="s2-en" checked={!scene2Spatial.enabled} onChange={() => updateSpatialState('scene2', { enabled: false })} /> Disabled</label>
            <span style={{ margin: '0 4px' }}>Objects in</span>
            <label><input type="radio" name="s2-lg" checked={scene2Spatial.logic === 'AND'} onChange={() => updateSpatialState('scene2', { logic: 'AND' })} /> AND</label>
            <label><input type="radio" name="s2-lg" checked={scene2Spatial.logic === 'OR'} onChange={() => updateSpatialState('scene2', { logic: 'OR' })} /> OR</label>
          </div>
        </div>

        <div className="adv-canvas-viewport">
          <div className="adv-dashed-grid"></div>
          {scene2Spatial.selectedObjects.length > 0 && (
            <div className="canvas-tags-overlay">
              {scene2Spatial.selectedObjects.map((tag) => (
                <span key={tag} className="tag-item" onClick={() => toggleObjectInSpatial('scene2', tag)}>
                  {tag} ✕
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="adv-box-footer">
          <textarea
            className="adv-max-obj-input"
            value={scene2Spatial.maxObjectsText}
            onChange={(e) => updateSpatialState('scene2', { maxObjectsText: e.target.value })}
            placeholder="Max Obj: e.g.: 2 person 3 car 0 dog, means at most 2 persons, 3 cars, no dogs"
          />
          <div className="adv-actions-stack">
            <button type="button" className="adv-btn-tool red" onClick={() => clearSpatialState('scene2')} title="Clear"><Trash2 size={16} /></button>
            <button type="button" className="adv-btn-tool green" onClick={() => clearSpatialState('scene2')} title="Reset"><RotateCcw size={16} /></button>
          </div>
        </div>
      </div>

      {/* 5. Image URL / Upload */}
      <div className="adv-image-search-section">
        <div className="adv-image-header">
          <span>Paste image URL</span>
          <button type="button" className="adv-upload-btn" onClick={handleUploadImage}><Upload size={13} /> Upload an image</button>
        </div>
        <div className="adv-url-input-wrap">
          <input
            type="text"
            placeholder="https://..."
            value={visualQuery.imageUrl}
            onChange={(e) => updateVisualQuery({ imageUrl: e.target.value })}
          />
          <button type="button" className="adv-lens-btn" onClick={executeSearch} title="Search by image"><Search size={16} /></button>
        </div>
        <div className="adv-eval-row">
          <span>EvaluationID</span>
          <select defaultValue="ERROR">
            <option value="ERROR">ERROR</option>
            <option value="SUBMISSION">SUBMISSION</option>
          </select>
        </div>
      </div>
    </aside>
  );
}