import { create } from 'zustand';
import type { KeyframeResult, SubmissionItem } from '../types/competition';

interface SubmissionState {
  items: SubmissionItem[];
  maxItems: number;
  addItem: (item: KeyframeResult) => void;
  removeItem: (keyframeId: string) => void;
  reorderItems: (activeKeyframeId: string, overKeyframeId: string) => void;
  clearAll: () => void;
}

export const useSubmissionStore = create<SubmissionState>((set) => ({
  items: [],
  maxItems: 100,

  addItem: (item: KeyframeResult) =>
    set((state) => {
      if (state.items.some((i) => i.keyframe_id === item.keyframe_id)) return state;
      if (state.items.length >= state.maxItems) {
        alert('Khay nộp bài đã đầy (tối đa 100 kết quả)!');
        return state;
      }
      const newItem: SubmissionItem = {
        keyframe_id: item.keyframe_id,
        video_id: item.video_id,
        frame_id: item.frame_id,
        shot_id: item.shot_id,
        timestamp: item.timestamp || '00:02:00',
        score: item.score || 0.95,
        timestamp_sec: item.timestamp_sec,
        thumbnail_url: item.thumbnail_url,
        video_url: item.video_url,
        objects: item.objects,
      };
      return { items: [...state.items, newItem] };
    }),

  removeItem: (keyframeId: string) =>
    set((state) => ({
      items: state.items.filter((i) => i.keyframe_id !== keyframeId),
    })),

  reorderItems: (activeKeyframeId: string, overKeyframeId: string) =>
    set((state) => {
      const oldIndex = state.items.findIndex((i) => i.keyframe_id === activeKeyframeId);
      const newIndex = state.items.findIndex((i) => i.keyframe_id === overKeyframeId);
      if (oldIndex === -1 || newIndex === -1) return state;

      const newItems = [...state.items];
      const [removed] = newItems.splice(oldIndex, 1);
      newItems.splice(newIndex, 0, removed);

      return { items: newItems };
    }),

  clearAll: () => set({ items: [] }),
}));
