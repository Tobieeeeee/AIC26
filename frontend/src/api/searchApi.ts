import axios from 'axios';
import { apiClient } from './client';
import type {
  KeyframeResult,
  LogicMode,
  RetrievalMode,
  TaskMode,
} from '../types/competition';

export interface SearchPayload {
  task_mode: TaskMode;
  query: string;
  retrieval_mode: RetrievalMode;
  sim_reorder: boolean;
  spatial_filter?: {
    enabled: boolean;
    logic: LogicMode;
    objects: string[];
    colors: string[];
    max_objects_text?: string;
  };
  visual_query?: {
    image_url: string;
  };
  top_k?: number;
}

interface SearchResponse {
  results?: unknown[];
}

const formatTimestamp = (value: unknown): string => {
  if (typeof value === 'string') return value;
  if (typeof value !== 'number' || !Number.isFinite(value)) return '';

  const totalSeconds = Math.max(0, Math.round(value));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [hours, minutes, seconds]
    .map((part) => String(part).padStart(2, '0'))
    .join(':');
};

const normalizeResult = (raw: unknown): KeyframeResult => {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Search response contained an invalid result.');
  }

  const item = raw as Record<string, unknown>;
  const keyframeId = typeof item.keyframe_id === 'string' ? item.keyframe_id : '';
  const videoId = typeof item.video_id === 'string' ? item.video_id : '';
  const frameValue = typeof item.frame_id === 'number' ? item.frame_id : Number(item.frame_id);

  if (!keyframeId || !videoId || !Number.isFinite(frameValue)) {
    throw new Error('Search response is missing canonical keyframe identity.');
  }

  const timestampValue = item.timestamp;
  const timestampSec =
    typeof timestampValue === 'number'
      ? timestampValue
      : typeof item.timestamp_sec === 'number'
        ? item.timestamp_sec
        : undefined;

  return {
    keyframe_id: keyframeId,
    video_id: videoId,
    frame_id: frameValue,
    shot_id: typeof item.shot_id === 'string' ? item.shot_id : undefined,
    timestamp: formatTimestamp(timestampValue ?? timestampSec),
    timestamp_sec: timestampSec,
    thumbnail_url:
      typeof item.thumbnail_url === 'string'
        ? item.thumbnail_url
        : typeof item.image_path === 'string'
          ? item.image_path
          : undefined,
    video_url: typeof item.video_url === 'string' ? item.video_url : undefined,
    score: typeof item.score === 'number' ? item.score : 0,
    objects: Array.isArray(item.objects)
      ? item.objects.filter((object): object is string => typeof object === 'string')
      : undefined,
    rank: typeof item.rank === 'number' ? item.rank : undefined,
    source: typeof item.source === 'string' ? item.source : undefined,
    ocr_text: typeof item.ocr_text === 'string' ? item.ocr_text : undefined,
    asr_text: typeof item.asr_text === 'string' ? item.asr_text : undefined,
  };
};

const errorMessage = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    if (error.response?.status === 404) {
      return 'Unified search is not available yet. Please start Task06A backend orchestration.';
    }
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (error.code === 'ECONNABORTED') return 'Search request timed out.';
    if (!error.response) return 'Unable to reach the search backend.';
  }
  return error instanceof Error ? error.message : 'Search failed.';
};

export const searchApi = {
  search: async (payload: SearchPayload): Promise<KeyframeResult[]> => {
    try {
      const response = await apiClient.post<SearchResponse>('/search', payload);
      if (!Array.isArray(response.data.results)) {
        throw new Error('Search response did not contain a results array.');
      }
      return response.data.results.map(normalizeResult);
    } catch (error) {
      const message = errorMessage(error);
      console.error('Search request failed:', message);
      throw new Error(message);
    }
  },
};
