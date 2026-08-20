export type TaskMode = 'kis' | 'trake' | 'qa';
export type RetrievalMode = 'semantic' | 'lexical' | 'object' | 'hybrid';
export type LogicMode = 'AND' | 'OR';
export type AdvancedPanelId = 'scene1' | 'scene2';

export interface KeyframeResult {
  keyframe_id: string;
  video_id: string;
  frame_id: number;
  shot_id?: string;
  timestamp: string;
  timestamp_sec?: number;
  thumbnail_url?: string;
  video_url?: string;
  score: number;
  objects?: string[];
  rank?: number;
  source?: string;
  ocr_text?: string;
  asr_text?: string;
}

export interface SubmissionItem {
  keyframe_id: string;
  video_id: string;
  frame_id: number;
  shot_id?: string;
  timestamp: string;
  score: number;
  timestamp_sec?: number;
  thumbnail_url?: string;
  video_url?: string;
  objects?: string[];
}

export interface SpatialCanvasState {
  enabled: boolean;
  logic: LogicMode;
  selectedObjects: string[];
  selectedColors: string[];
  maxObjectsText: string;
}

export interface VisualQueryState {
  mode: 'url' | 'upload';
  imageUrl: string;
  filename: string;
  evaluationId: string;
}
