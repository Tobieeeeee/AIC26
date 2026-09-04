export type TaskMode = "KIS" | "TRAKE" | "Q&A";

export type ModelEngine = "ClipVideo" | "ClipLAION" | "Aladin" | "use all";

export type Language = "Vietnamese" | "English" | "Italian" | "French" | "Japanese" | "Korean";

export type SpatialLogic = "AND" | "OR";

export type SpatialFilter = {
  enabled: boolean;
  logic: SpatialLogic;
  objects: string[];
  colors: string[];
};

export type VisualQuery = {
  image_url: string;
};

export type UiSearchRequest = {
  task_mode: TaskMode;
  query: string;
  model_engine: ModelEngine;
  sim_reorder: boolean;
  spatial_filter?: SpatialFilter;
  visual_query?: VisualQuery;
  top_k: number;
  question?: string;
  event_description?: string;
  trake_events?: string[];
  gap_phrases?: string[];
};

export type UiKeyframeResult = {
  id: string;
  video_id: string;
  frame_id: number;
  timestamp: string;
  timestamp_sec: number;
  thumbnail_url: string;
  video_url: string;
  score: number;
  objects: string[];
  answer: string | null;
  frames: Array<number | null> | null;
};

export type UiSearchResponse = {
  status: "ok";
  total_results: number;
  results: UiKeyframeResult[];
};

export type VideoSummaryKeyframe = {
  keyframe_id: string;
  frame_id: number | null;
  timestamp: string | null;
  timestamp_sec: number | null;
  thumbnail_url: string;
  video_url: string;
  mapped: boolean;
};

export type VideoSummaryResponse = {
  video_id: string;
  keyframe_count: number;
  mapped_count: number;
  results: VideoSummaryKeyframe[];
};
