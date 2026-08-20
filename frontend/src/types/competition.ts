export type TaskMode = 'KIS' | 'TRAKE' | 'Q&A';
export type ModelEngine = 'use all' | 'ClipVideo' | 'ClipLAION' | 'Aladin';
export type LogicMode = 'AND' | 'OR';
export type AdvancedPanelId = 'scene1' | 'scene2';
export type RetrievalMethod = 'ocr' | 'asr' | 'clip' | 'fusion' | 'spatial';

export interface KeyframeResult {
  id: string;
  video_id: string;
  frame_id: number;
  timestamp: string;
  timestamp_sec: number;
  thumbnail_url: string;
  video_url: string;
  score: number;
  objects?: string[];
}

export interface SubmissionItem {
  id: string;
  video_id: string;
  frame_id: number;
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