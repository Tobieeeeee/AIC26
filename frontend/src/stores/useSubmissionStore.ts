import { create } from 'zustand';
import type { KeyframeResult, SubmissionItem } from '../types/competition';

interface SubmissionState {
  items: SubmissionItem[];
  maxItems: number;
  addItem: (item: KeyframeResult) => void;
  removeItem: (id: string) => void;
  reorderItems: (activeId: string, overId: string) => void;
  clearAll: () => void;
}

export const useSubmissionStore = create<SubmissionState>((set) => ({
  items: [],
  maxItems: 100,

  addItem: (item: KeyframeResult) =>
    set((state) => {
      if (state.items.some((i) => i.id === item.id)) return state;
      if (state.items.length >= state.maxItems) {
        alert('Khay nộp bài đã đầy (tối đa 100 kết quả)!');
        return state;
      }
      const newItem: SubmissionItem = {
        id: item.id,
        video_id: item.video_id,
        frame_id: item.frame_id,
        timestamp: item.timestamp || '00:02:00',
        score: item.score || 0.95,
      };
      return { items: [...state.items, newItem] };
    }),

  removeItem: (id: string) =>
    set((state) => ({
      items: state.items.filter((i) => i.id !== id),
    })),

  reorderItems: (activeId: string, overId: string) =>
    set((state) => {
      const oldIndex = state.items.findIndex((i) => i.id === activeId);
      const newIndex = state.items.findIndex((i) => i.id === overId);
      if (oldIndex === -1 || newIndex === -1) return state;

      const newItems = [...state.items];
      const [removed] = newItems.splice(oldIndex, 1);
      newItems.splice(newIndex, 0, removed);

      return { items: newItems };
    }),

  clearAll: () => set({ items: [] }),
}));