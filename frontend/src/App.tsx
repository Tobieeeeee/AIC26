import {
  Activity,
  Armchair,
  Bike,
  Bird,
  BookOpen,
  Cake,
  Car,
  Cat,
  ChevronLeft,
  ChevronRight,
  Check,
  Coffee,
  Dog,
  Eraser,
  Flower2,
  Footprints,
  Glasses,
  Grid3X3,
  Home,
  Hourglass,
  Languages,
  Laptop,
  Loader2,
  Music,
  Navigation,
  Plane,
  Play,
  Search,
  Ship,
  Smile,
  Sofa,
  Sparkles,
  Smartphone,
  Trash2,
  Trees,
  Trophy,
  Truck,
  Tv,
  Umbrella,
  Undo2,
  User,
  UserCheck,
  UserRound,
  Wine,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, FormEvent } from "react";
import type { LucideIcon } from "lucide-react";
import { assetUrl, getVideoSummary, search } from "./api";
import type {
  Language,
  ModelEngine,
  SpatialFilter,
  SpatialLogic,
  TaskMode,
  UiKeyframeResult,
  VideoSummaryKeyframe,
  VideoSummaryResponse,
} from "./types";

const taskModes: TaskMode[] = ["KIS", "TRAKE", "Q&A"];
const modelEngines: ModelEngine[] = ["ClipVideo", "ClipLAION", "Aladin", "use all"];
const languages: Language[] = ["Vietnamese", "English", "Italian", "French", "Japanese", "Korean"];
const sessionOptions = ["ERROR", "SESSION_01", "DEFAULT"];
const summaryPageSize = 48;

const objectOptions: Array<{ label: string; value: string; Icon: LucideIcon }> = [
  { label: "person", value: "person", Icon: User },
  { label: "man", value: "man", Icon: UserCheck },
  { label: "woman", value: "woman", Icon: UserRound },
  { label: "face", value: "face", Icon: Smile },
  { label: "music", value: "music", Icon: Music },
  { label: "sports", value: "sports", Icon: Trophy },
  { label: "tennis", value: "tennis", Icon: Activity },
  { label: "skateboard", value: "skateboard", Icon: Footprints },
  { label: "glasses", value: "glasses", Icon: Glasses },
  { label: "bicycle", value: "bicycle", Icon: Bike },
  { label: "motorcycle", value: "motorcycle", Icon: Bike },
  { label: "car", value: "car", Icon: Car },
  { label: "truck", value: "truck", Icon: Truck },
  { label: "boat", value: "boat", Icon: Ship },
  { label: "parachute", value: "parachute", Icon: Navigation },
  { label: "plane", value: "plane", Icon: Plane },
  { label: "bench", value: "bench", Icon: Armchair },
  { label: "office chair", value: "office chair", Icon: Armchair },
  { label: "couch", value: "couch", Icon: Sofa },
  { label: "house", value: "house", Icon: Home },
  { label: "umbrella", value: "umbrella", Icon: Umbrella },
  { label: "wine glass", value: "wine glass", Icon: Wine },
  { label: "cup", value: "cup", Icon: Coffee },
  { label: "cake", value: "cake", Icon: Cake },
  { label: "phone", value: "phone", Icon: Smartphone },
  { label: "tv", value: "tv", Icon: Tv },
  { label: "laptop", value: "laptop", Icon: Laptop },
  { label: "book", value: "book", Icon: BookOpen },
  { label: "cat", value: "cat", Icon: Cat },
  { label: "dog", value: "dog", Icon: Dog },
  { label: "horse", value: "horse", Icon: Sparkles },
  { label: "bird", value: "bird", Icon: Bird },
  { label: "flower", value: "flower", Icon: Flower2 },
  { label: "tree", value: "tree", Icon: Trees },
];

const colorOptions = [
  { label: "BLACK", value: "black", hex: "#05070b" },
  { label: "BLUE", value: "blue", hex: "#2563eb" },
  { label: "BROWN", value: "brown", hex: "#7c3f1d" },
  { label: "GREEN", value: "green", hex: "#16a34a" },
  { label: "GREY", value: "grey", hex: "#71717a" },
  { label: "ORANGE", value: "orange", hex: "#f97316" },
  { label: "PINK", value: "pink", hex: "#ec4899" },
  { label: "PURPLE", value: "purple", hex: "#7c3aed" },
  { label: "RED", value: "red", hex: "#dc2626" },
  { label: "WHITE", value: "white", hex: "#ffffff" },
  { label: "YELLOW", value: "yellow", hex: "#eab308" },
];

const fixtureResults: UiKeyframeResult[] = Array.from({ length: 12 }, (_, index) => {
  const videoNumber = Math.floor(index / 4) + 1;
  const frame = 1240 + index * 18;
  return {
    id: `fixture-${index + 1}`,
    video_id: `L0${videoNumber}_V${String(index + 1).padStart(3, "0")}`,
    frame_id: frame,
    timestamp: `00:0${Math.floor(index / 2) + 2}:${String((index * 7) % 60).padStart(2, "0")}`,
    timestamp_sec: 120 + index * 7,
    thumbnail_url: "",
    video_url: "",
    score: Math.max(0.41, 0.98 - index * 0.035),
    objects: index % 2 === 0 ? ["person", "car", "tree"] : ["person", "phone"],
    answer: null,
    frames: null,
  };
});

type SceneQuery = {
  text: string;
  translate: boolean;
  language: Language;
  engine: ModelEngine;
};

type SpatialScene = SpatialFilter & {
  maxObjects: string;
};

type VisualQueryState = {
  tab: "url" | "upload";
  imageUrl: string;
  fileName: string;
  session: string;
};

type StatusValue = "Enabled" | "Disabled";

function emptySpatialScene(): SpatialScene {
  return {
    enabled: true,
    logic: "AND",
    objects: [],
    colors: [],
    maxObjects: "",
  };
}

function scorePercent(score: number): string {
  return `${Math.round(score * 100)}%`;
}

function createModePayload(sceneOne: SceneQuery, sceneTwo: SceneQuery, taskMode: TaskMode) {
  if (taskMode === "TRAKE" && sceneTwo.text.trim()) {
    const trakeEvents = [sceneOne.text.trim(), sceneTwo.text.trim()];
    return {
      trake_events: trakeEvents,
      gap_phrases: Array(trakeEvents.length - 1).fill(""),
    };
  }

  if (taskMode === "Q&A") {
    return {
      event_description: sceneOne.text.trim(),
      question: sceneTwo.text.trim() || sceneOne.text.trim(),
    };
  }

  return {};
}

function compactSpatialFilter(sceneOneSpatial: SpatialScene, sceneTwoSpatial: SpatialScene): SpatialFilter {
  return {
    enabled: sceneOneSpatial.enabled || sceneTwoSpatial.enabled,
    logic: sceneOneSpatial.logic,
    objects: [...new Set([...sceneOneSpatial.objects, ...sceneTwoSpatial.objects])],
    colors: [...new Set([...sceneOneSpatial.colors, ...sceneTwoSpatial.colors])],
  };
}

function toggleListValue<T extends "objects" | "colors">(state: SpatialScene, key: T, value: string): SpatialScene {
  const exists = state[key].includes(value);
  return {
    ...state,
    [key]: exists ? state[key].filter((item) => item !== value) : [...state[key], value],
  };
}

export function App() {
  const [taskMode, setTaskMode] = useState<TaskMode>("KIS");
  const [simReorder, setSimReorder] = useState(true);
  const [advanced, setAdvanced] = useState(false);
  const [sceneOne, setSceneOne] = useState<SceneQuery>({
    text: "con lan",
    translate: true,
    language: "Vietnamese",
    engine: "use all",
  });
  const [sceneTwo, setSceneTwo] = useState<SceneQuery>({
    text: "",
    translate: true,
    language: "Vietnamese",
    engine: "use all",
  });
  const [sceneOneSpatial, setSceneOneSpatial] = useState<SpatialScene>(emptySpatialScene);
  const [sceneTwoSpatial, setSceneTwoSpatial] = useState<SpatialScene>(emptySpatialScene);
  const [visualQuery, setVisualQuery] = useState<VisualQueryState>({
    tab: "url",
    imageUrl: "",
    fileName: "No file chosen",
    session: "DEFAULT",
  });
  const [topK, setTopK] = useState(24);
  const [results, setResults] = useState<UiKeyframeResult[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [submission, setSubmission] = useState<UiKeyframeResult[]>([]);
  const [openVideo, setOpenVideo] = useState<UiKeyframeResult | null>(null);
  const [summaryVideoId, setSummaryVideoId] = useState("");
  const [searching, setSearching] = useState(false);
  const [status, setStatus] = useState("Ready to search real keyframes");

  const selectedResult = useMemo(
    () => results.find((result) => result.id === selectedId) ?? results[0],
    [results, selectedId],
  );

  async function runSearch(event?: FormEvent) {
    event?.preventDefault();
    const query = sceneOne.text.trim();
    if (!query) {
      setStatus("Nhap mo ta can tim truoc khi search.");
      return;
    }

    setSearching(true);
    setStatus("Searching backend...");
    try {
      const response = await search({
        task_mode: taskMode,
        query,
        model_engine: sceneOne.engine,
        sim_reorder: simReorder,
        spatial_filter: advanced ? compactSpatialFilter(sceneOneSpatial, sceneTwoSpatial) : undefined,
        visual_query: visualQuery.imageUrl.trim() ? { image_url: visualQuery.imageUrl.trim() } : undefined,
        top_k: Math.min(Math.max(topK, 1), 100),
        ...createModePayload(sceneOne, sceneTwo, taskMode),
      });
      const nextResults = response.results;
      setResults(nextResults);
      setSelectedId(nextResults[0]?.id ?? "");
      setStatus(`API status: Online (${response.total_results} results)`);
    } catch (error) {
      setResults([]);
      setSelectedId("");
      setStatus(error instanceof Error ? `Backend unavailable: ${error.message}` : "Backend unavailable");
    } finally {
      setSearching(false);
    }
  }

  function addSubmission(result: UiKeyframeResult) {
    setSubmission((current) => (current.some((item) => item.id === result.id) ? current : [...current, result]));
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="wordmark">OKCHOKCHOK</div>
          <div>
            <h1>Okchokchok</h1>
            <p>Video retrieval operator console</p>
          </div>
        </div>

        <div className="header-controls" aria-label="Search controls">
          <RadioGroup<TaskMode> label="Task mode" value={taskMode} values={taskModes} onChange={setTaskMode} />
          <label className="inline-check">
            <input type="checkbox" checked={simReorder} onChange={(event) => setSimReorder(event.target.checked)} />
            Sim Reorder
          </label>
          <label className="advanced-switch">
            <span>ADVANCED</span>
            <input type="checkbox" checked={advanced} onChange={(event) => setAdvanced(event.target.checked)} />
            <span className="switch-face">{advanced ? "ON" : "OFF"}</span>
          </label>
        </div>
      </header>

      <section className={advanced ? "operator-layout advanced" : "operator-layout"}>
        {advanced ? (
          <aside className="spatial-panel">
            <ObjectPalette
              selected={sceneOneSpatial.objects}
              onToggle={(value) => setSceneOneSpatial((current) => toggleListValue(current, "objects", value))}
            />
            <ColorPalette
              selected={sceneOneSpatial.colors}
              onToggle={(value) => setSceneOneSpatial((current) => toggleListValue(current, "colors", value))}
            />
            <SpatialCanvas
              title="Object Spatial Scene 1"
              tone="red"
              state={sceneOneSpatial}
              onChange={setSceneOneSpatial}
            />
            <SpatialCanvas
              title="Next Scene Spatial"
              tone="green"
              state={sceneTwoSpatial}
              onChange={setSceneTwoSpatial}
            />
            <VisualQueryPanel visualQuery={visualQuery} onChange={setVisualQuery} onSearch={() => void runSearch()} />
          </aside>
        ) : null}

        <section className="search-workspace">
          <form className={advanced ? "query-stack" : "standard-query"} onSubmit={runSearch}>
            <SceneQueryBar
              title="Scene 1"
              tone="red"
              scene={sceneOne}
              placeholder="Describe the scene you are looking for..."
              onChange={setSceneOne}
            />
            {advanced ? (
              <SceneQueryBar
                title="Scene 2"
                tone="green"
                scene={sceneTwo}
                placeholder="Describe the next scene (optional)..."
                icon="hourglass"
                onChange={setSceneTwo}
              />
            ) : null}
            <div className="query-actions">
              <label className="topk-field">
                Top K
                <input
                  type="number"
                  min={1}
                  max={100}
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </label>
              <button className="search-button" type="submit" disabled={searching}>
                {searching ? <Loader2 className="spin" size={18} /> : <Search size={19} />}
              </button>
            </div>
          </form>

          <section className="results-section">
            <div className="section-heading">
              <div>
                <h2>
                  <Grid3X3 size={17} /> Ranked Keyframes
                </h2>
                <p>{status}</p>
              </div>
              <button
                className="fixture-button"
                type="button"
                onClick={() => {
                  const demoResults = fixtureResults.slice(0, topK);
                  setResults(demoResults);
                  setSelectedId(demoResults[0]?.id ?? "");
                  setStatus("Demo fixtures loaded (not backend results)");
                }}
              >
                Load Demo
              </button>
            </div>
            <div className="result-grid">
              {results.map((result, index) => (
                <button
                  key={result.id}
                  className={result.id === selectedResult?.id ? "result-card selected" : "result-card"}
                  type="button"
                  title={result.video_url ? `Open ${result.video_id} at ${result.timestamp}` : undefined}
                  onClick={() => {
                    setSelectedId(result.id);
                    if (result.video_url) {
                      setOpenVideo(result);
                    }
                  }}
                >
                  <span className="rank">#{index + 1}</span>
                  <span className="result-media">
                    {result.thumbnail_url ? (
                      <img src={assetUrl(result.thumbnail_url)} alt={`${result.video_id} frame ${result.frame_id}`} />
                    ) : (
                      <span className="thumb-placeholder" />
                    )}
                    {result.video_url ? (
                      <span className="result-play" aria-hidden="true">
                        <Play size={22} fill="currentColor" />
                      </span>
                    ) : null}
                  </span>
                  <span className="result-copy">
                    <strong>{result.video_id}</strong>
                    <span>Frame {result.frame_id}</span>
                    <span>{scorePercent(result.score)} similarity</span>
                  </span>
                  <span
                    className="add-button"
                    role="button"
                    tabIndex={0}
                    onClick={(event) => {
                      event.stopPropagation();
                      addSubmission(result);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        event.stopPropagation();
                        addSubmission(result);
                      }
                    }}
                  >
                    + Add
                  </span>
                </button>
              ))}
              {!results.length && !searching ? (
                <p className="empty-results">No keyframes to display. Run a search or check the backend status above.</p>
              ) : null}
            </div>
          </section>
        </section>

        <aside className="right-rail">
          <PreviewPanel
            result={selectedResult}
            onOpen={setOpenVideo}
            onOpenSummary={(videoId) => setSummaryVideoId(videoId)}
          />
          <SubmissionTray
            items={submission}
            onClear={() => setSubmission([])}
            onRemove={(id) => setSubmission((current) => current.filter((item) => item.id !== id))}
          />
        </aside>
      </section>
      {openVideo ? <VideoModal result={openVideo} onClose={() => setOpenVideo(null)} /> : null}
      {summaryVideoId ? <VideoSummaryModal videoId={summaryVideoId} onClose={() => setSummaryVideoId("")} /> : null}
    </main>
  );
}

type RadioGroupProps<T extends string> = {
  label: string;
  value: T;
  values: T[];
  onChange: (value: T) => void;
};

function RadioGroup<T extends string>({ label, value, values, onChange }: RadioGroupProps<T>) {
  return (
    <fieldset className="radio-group">
      <legend>{label}</legend>
      {values.map((item) => (
        <label key={item}>
          <input type="radio" checked={item === value} onChange={() => onChange(item)} />
          {item}
        </label>
      ))}
    </fieldset>
  );
}

type SceneQueryBarProps = {
  title: string;
  tone: "red" | "green";
  scene: SceneQuery;
  placeholder: string;
  icon?: "search" | "hourglass";
  onChange: (scene: SceneQuery) => void;
};

function SceneQueryBar({ title, tone, scene, placeholder, icon = "search", onChange }: SceneQueryBarProps) {
  const Icon = icon === "hourglass" ? Hourglass : Search;

  return (
    <section className={`scene-query ${tone}`}>
      <label className="scene-input">
        <span>{title}</span>
        <div className="input-shell">
          <Icon size={20} />
          <input
            value={scene.text}
            onChange={(event) => onChange({ ...scene, text: event.target.value })}
            placeholder={placeholder}
          />
        </div>
      </label>

      <div className="translation-row">
        <label className="inline-check">
          <input
            type="checkbox"
            checked={scene.translate}
            onChange={(event) => onChange({ ...scene, translate: event.target.checked })}
          />
          <Languages size={18} />
          Translate
        </label>
        <select value={scene.language} onChange={(event) => onChange({ ...scene, language: event.target.value as Language })}>
          {languages.map((language) => (
            <option key={language}>{language}</option>
          ))}
        </select>
        <RadioGroup<ModelEngine>
          label="Model engine"
          value={scene.engine}
          values={modelEngines}
          onChange={(engine) => onChange({ ...scene, engine })}
        />
      </div>
    </section>
  );
}

type ObjectPaletteProps = {
  selected: string[];
  onToggle: (value: string) => void;
};

function ObjectPalette({ selected, onToggle }: ObjectPaletteProps) {
  return (
    <section className="palette-block">
      <h2>Object Palette</h2>
      <div className="object-grid">
        {objectOptions.map(({ label, value, Icon }) => (
          <button
            key={value}
            className={selected.includes(value) ? "object-icon active" : "object-icon"}
            type="button"
            title={label}
            aria-label={label}
            onClick={() => onToggle(value)}
          >
            <Icon size={20} strokeWidth={2.2} />
          </button>
        ))}
      </div>
    </section>
  );
}

type ColorPaletteProps = {
  selected: string[];
  onToggle: (value: string) => void;
};

function ColorPalette({ selected, onToggle }: ColorPaletteProps) {
  return (
    <section className="palette-block">
      <h2>Color Palette</h2>
      <div className="swatch-row">
        {colorOptions.map((color) => (
          <button
            key={color.value}
            className={selected.includes(color.value) ? "swatch active" : "swatch"}
            style={{ "--swatch": color.hex } as CSSProperties}
            type="button"
            onClick={() => onToggle(color.value)}
          >
            <span />
            <strong>{color.label}</strong>
          </button>
        ))}
      </div>
    </section>
  );
}

type SpatialCanvasProps = {
  title: string;
  tone: "red" | "green";
  state: SpatialScene;
  onChange: (state: SpatialScene) => void;
};

function SpatialCanvas({ title, tone, state, onChange }: SpatialCanvasProps) {
  return (
    <section className={`spatial-canvas ${tone}`}>
      <div className="canvas-header">
        <h2>
          <Grid3X3 size={16} /> {title}
        </h2>
        <RadioGroup<StatusValue>
          label="Status"
          value={state.enabled ? "Enabled" : "Disabled"}
          values={["Enabled", "Disabled"]}
          onChange={(value) => onChange({ ...state, enabled: value === "Enabled" })}
        />
        <RadioGroup<SpatialLogic> label="Objects in" value={state.logic} values={["AND", "OR"]} onChange={(logic) => onChange({ ...state, logic })} />
      </div>
      <div className="grid-canvas">
        {Array.from({ length: 9 }, (_, index) => (
          <button key={index} type="button" aria-label={`grid cell ${index + 1}`} />
        ))}
        <span>Max Objects Number</span>
      </div>
      <textarea
        value={state.maxObjects}
        onChange={(event) => onChange({ ...state, maxObjects: event.target.value })}
        placeholder="Max Obj: e.g.: 2 person 3 car 0 dog, means at most 2 persons, 3 cars, no dogs"
        rows={3}
      />
      <div className="canvas-actions">
        <button className="danger" type="button" onClick={() => onChange({ ...emptySpatialScene(), enabled: state.enabled })}>
          <Trash2 size={17} />
        </button>
        <button className="success" type="button" onClick={() => onChange(emptySpatialScene())}>
          <Undo2 size={17} />
        </button>
      </div>
    </section>
  );
}

type VisualQueryPanelProps = {
  visualQuery: VisualQueryState;
  onChange: (state: VisualQueryState) => void;
  onSearch: () => void;
};

function VisualQueryPanel({ visualQuery, onChange, onSearch }: VisualQueryPanelProps) {
  return (
    <section className="visual-panel">
      <div className="visual-tabs">
        <button
          className={visualQuery.tab === "url" ? "active" : ""}
          type="button"
          onClick={() => onChange({ ...visualQuery, tab: "url" })}
        >
          Paste image URL
        </button>
        <button
          className={visualQuery.tab === "upload" ? "active" : ""}
          type="button"
          onClick={() => onChange({ ...visualQuery, tab: "upload" })}
        >
          Upload an image
        </button>
      </div>
      {visualQuery.tab === "url" ? (
        <input
          value={visualQuery.imageUrl}
          onChange={(event) => onChange({ ...visualQuery, imageUrl: event.target.value })}
          placeholder="https://example.com/keyframe.jpg"
        />
      ) : (
        <label className="file-picker">
          <input
            type="file"
            accept="image/*"
            onChange={(event) => onChange({ ...visualQuery, fileName: event.target.files?.[0]?.name || "No file chosen" })}
          />
          <span>{visualQuery.fileName}</span>
        </label>
      )}
      <div className="visual-actions">
        <button type="button" onClick={onSearch}>
          <Search size={18} />
        </button>
        <label>
          EvaluationID
          <select value={visualQuery.session} onChange={(event) => onChange({ ...visualQuery, session: event.target.value })}>
            {sessionOptions.map((session) => (
              <option key={session}>{session}</option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

type PreviewPanelProps = {
  result?: UiKeyframeResult;
  onOpen: (result: UiKeyframeResult) => void;
  onOpenSummary: (videoId: string) => void;
};

function PreviewPanel({ result, onOpen, onOpenSummary }: PreviewPanelProps) {
  return (
    <section className="side-panel preview-panel">
      <h2>Preview</h2>
      <button className="preview-stage" type="button" disabled={!result?.video_url} onClick={() => result && onOpen(result)}>
        {result?.thumbnail_url ? <img src={assetUrl(result.thumbnail_url)} alt={`${result.video_id} frame`} /> : <span>TV</span>}
        <strong>{result?.video_url ? "Click to open video player" : "Video unavailable"}</strong>
      </button>
      {result ? (
        <dl>
          <dt>video_id</dt>
          <dd>{result.video_id}</dd>
          <dt>frame_id</dt>
          <dd>{result.frame_id}</dd>
          <dt>timestamp</dt>
          <dd>{result.timestamp}</dd>
          <dt>objects</dt>
          <dd>{result.objects.join(", ") || "unknown"}</dd>
        </dl>
      ) : null}
      <button
        className="summary-button"
        type="button"
        disabled={!result?.video_id}
        onClick={() => result && onOpenSummary(result.video_id)}
      >
        <Grid3X3 size={17} />
        Video Summary
      </button>
    </section>
  );
}

type VideoModalProps = {
  result: UiKeyframeResult;
  onClose: () => void;
};

function VideoModal({ result, onClose }: VideoModalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  function seekToResult() {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, result.timestamp_sec);
    }
  }

  useEffect(() => {
    seekToResult();
  }, [result]);

  return (
    <div className="video-modal" role="dialog" aria-modal="true" aria-label={`${result.video_id} video player`}>
      <div className="video-dialog">
        <header>
          <div>
            <h2>{result.video_id}</h2>
            <p>
              Frame {result.frame_id} at {result.timestamp}
            </p>
          </div>
          <button type="button" title="Close" aria-label="Close video" onClick={onClose}>
            <X size={19} />
          </button>
        </header>
        <video ref={videoRef} controls autoPlay src={assetUrl(result.video_url)} onLoadedMetadata={seekToResult} />
      </div>
    </div>
  );
}

type VideoSummaryModalProps = {
  videoId: string;
  onClose: () => void;
};

function VideoSummaryModal({ videoId, onClose }: VideoSummaryModalProps) {
  const [summary, setSummary] = useState<VideoSummaryResponse | null>(null);
  const [selected, setSelected] = useState<VideoSummaryKeyframe | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [page, setPage] = useState(0);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError("");
    getVideoSummary(videoId)
      .then((response) => {
        if (!alive) {
          return;
        }
        setSummary(response);
        setSelected(response.results[0] ?? null);
        setPage(0);
      })
      .catch((reason) => {
        if (!alive) {
          return;
        }
        setError(reason instanceof Error ? reason.message : "Could not load video summary");
        setSummary(null);
        setSelected(null);
      })
      .finally(() => {
        if (alive) {
          setLoading(false);
        }
      });
    return () => {
      alive = false;
    };
  }, [videoId]);

  useEffect(() => {
    if (videoRef.current && typeof selected?.timestamp_sec === "number") {
      videoRef.current.currentTime = Math.max(0, selected.timestamp_sec);
    }
  }, [selected]);

  const videoUrl = selected?.video_url || summary?.results.find((item) => item.video_url)?.video_url || "";
  const pageCount = Math.max(1, Math.ceil((summary?.results.length ?? 0) / summaryPageSize));
  const pageStart = page * summaryPageSize;
  const visibleKeyframes = summary?.results.slice(pageStart, pageStart + summaryPageSize) ?? [];

  function seekToSelected() {
    if (videoRef.current && typeof selected?.timestamp_sec === "number") {
      videoRef.current.currentTime = Math.max(0, selected.timestamp_sec);
    }
  }

  return (
    <div className="video-modal summary-modal" role="dialog" aria-modal="true" aria-label={`${videoId} video summary`}>
      <div className="summary-dialog">
        <header>
          <div>
            <h2>Video {videoId} - Keyframes</h2>
            <p>{summary ? `${summary.keyframe_count} keyframes, ${summary.mapped_count} mapped` : "Loading keyframes"}</p>
          </div>
          <button type="button" title="Close" aria-label="Close video summary" onClick={onClose}>
            <X size={19} />
          </button>
        </header>

        {loading ? (
          <div className="summary-state">
            <Loader2 className="spin" size={22} /> Loading summary
          </div>
        ) : null}
        {error ? <div className="summary-state error">{error}</div> : null}

        {summary ? (
          <div className="summary-browser">
            <section className="summary-focus">
              {videoUrl ? (
                <video ref={videoRef} controls src={assetUrl(videoUrl)} onLoadedMetadata={seekToSelected} />
              ) : (
                <div className="summary-state">Video unavailable</div>
              )}
              {selected ? (
                <div className="summary-focus-copy">
                  <span>Selected keyframe</span>
                  <h3>{videoId}-{selected.keyframe_id}</h3>
                  <dl>
                    <dt>keyframe</dt>
                    <dd>{selected.keyframe_id}</dd>
                    <dt>frame_id</dt>
                    <dd>{selected.frame_id ?? "unmapped"}</dd>
                    <dt>timestamp</dt>
                    <dd>{selected.timestamp ?? "unknown"}</dd>
                  </dl>
                </div>
              ) : null}
            </section>

            <div className="summary-toolbar">
              <div>
                <strong>Keyframe browser</strong>
                <span>
                  {pageStart + 1}-{Math.min(pageStart + summaryPageSize, summary.results.length)} of {summary.results.length}
                </span>
              </div>
              <nav aria-label="Keyframe pages">
                <button type="button" title="Previous page" aria-label="Previous keyframe page" disabled={page === 0} onClick={() => setPage((current) => Math.max(0, current - 1))}>
                  <ChevronLeft size={18} />
                </button>
                <span>
                  Page {page + 1} / {pageCount}
                </span>
                <button type="button" title="Next page" aria-label="Next keyframe page" disabled={page + 1 >= pageCount} onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}>
                  <ChevronRight size={18} />
                </button>
              </nav>
            </div>

            <section className="summary-grid" aria-label="Video keyframes">
              {visibleKeyframes.map((item, index) => (
                <button
                  key={item.keyframe_id}
                  className={selected?.keyframe_id === item.keyframe_id ? "summary-tile active" : "summary-tile"}
                  type="button"
                  title={`${videoId}-${item.keyframe_id} at ${item.timestamp ?? "unknown"}`}
                  onClick={() => setSelected(item)}
                >
                  <span className="summary-image">
                    <img
                      src={assetUrl(item.thumbnail_url)}
                      alt={`${videoId} keyframe ${item.keyframe_id}`}
                      loading="lazy"
                      decoding="async"
                    />
                    <span className="summary-number">#{String(pageStart + index + 1).padStart(3, "0")}</span>
                    <span className="summary-play" aria-hidden="true">
                      <Play size={17} fill="currentColor" />
                    </span>
                  </span>
                  <span className="summary-tile-copy">
                    <strong>{videoId}-{item.keyframe_id}</strong>
                    <small>Frame {item.frame_id ?? "unmapped"}</small>
                    <time>{item.timestamp ?? "unknown"}</time>
                  </span>
                </button>
              ))}
            </section>
          </div>
        ) : null}
      </div>
    </div>
  );
}

type SubmissionTrayProps = {
  items: UiKeyframeResult[];
  onClear: () => void;
  onRemove: (id: string) => void;
};

function SubmissionTray({ items, onClear, onRemove }: SubmissionTrayProps) {
  return (
    <section className="side-panel submission-panel">
      <div className="submission-head">
        <h2>
          <Check size={18} /> Submission Tray
        </h2>
        <button type="button" onClick={onClear}>
          <Eraser size={16} />
          Clear
        </button>
      </div>
      <p>Selected {items.length} / 100 (Keo tha de doi thu tu)</p>
      {items.length ? (
        <ol>
          {items.map((item) => (
            <li key={item.id}>
              <span>
                {item.video_id} / {item.frame_id}
              </span>
              <button type="button" onClick={() => onRemove(item.id)}>
                Remove
              </button>
            </li>
          ))}
        </ol>
      ) : (
        <span className="empty-tray">Khay nop bai dang trong.</span>
      )}
    </section>
  );
}
