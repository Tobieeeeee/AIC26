import { apiClient } from './client';
import type { KeyframeResult, TaskMode, ModelEngine } from '../types/competition';

interface SearchPayload {
  task_mode: TaskMode;
  query: string;
  model_engine: ModelEngine;
  sim_reorder: boolean;
  spatial_filter?: {
    enabled: boolean;
    logic: 'AND' | 'OR';
    objects: string[];
    colors: string[];
  };
  visual_query?: {
    image_url: string;
  };
  top_k?: number;
}

interface SearchResponse {
  status: string;
  total_results: number;
  results: KeyframeResult[];
}

export const searchApi = {
  // Gửi request tìm kiếm lên FastAPI backend
  search: async (payload: SearchPayload): Promise<KeyframeResult[]> => {
    try {
      const response = await apiClient.post<SearchResponse>('/search', payload);
      return response.data.results || [];
    } catch (error) {
      console.error('API Search Error, fallback to mock/empty:', error);
      throw error;
    }
  },

  // (Tùy chọn) Gửi submit kết quả trực tiếp qua API nếu backend hỗ trợ
  submitResults: async (queryId: string, taskMode: TaskMode, items: { video_id: string; frame_id: number; answer?: string }[]) => {
    try {
      const response = await apiClient.post('/submit', {
        query_id: queryId,
        task_mode: taskMode,
        items,
      });
      return response.data;
    } catch (error) {
      console.error('API Submit Error:', error);
      throw error;
    }
  },
};