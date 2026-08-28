import { create } from 'zustand';
import { searchApi } from '../api/searchApi';
import type { KeyframeResult, TaskMode, ModelEngine } from '../types/competition';

const initialMockResults: KeyframeResult[] = [
  { id: '1', video_id: 'L01_V001', frame_id: 1240, timestamp: '00:02:00', timestamp_sec: 120, score: 0.98, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['person', 'car', 'tree'] },
  { id: '2', video_id: 'L01_V002', frame_id: 1258, timestamp: '00:02:05', timestamp_sec: 125, score: 0.95, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['person', 'motorcycle'] },
  { id: '3', video_id: 'L01_V003', frame_id: 1276, timestamp: '00:02:11', timestamp_sec: 131, score: 0.91, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['car', 'building'] },
  { id: '4', video_id: 'L01_V004', frame_id: 1294, timestamp: '00:02:18', timestamp_sec: 138, score: 0.88, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['person'] },
  { id: '5', video_id: 'L02_V005', frame_id: 1312, timestamp: '00:02:22', timestamp_sec: 142, score: 0.84, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['dog', 'tree'] },
  { id: '6', video_id: 'L02_V006', frame_id: 1330, timestamp: '00:02:27', timestamp_sec: 147, score: 0.81, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['bicycle'] },
  { id: '7', video_id: 'L02_V007', frame_id: 1348, timestamp: '00:02:35', timestamp_sec: 155, score: 0.77, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['boat', 'water'] },
  { id: '8', video_id: 'L02_V008', frame_id: 1366, timestamp: '00:02:40', timestamp_sec: 160, score: 0.74, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['truck'] },
  { id: '9', video_id: 'L03_V009', frame_id: 1384, timestamp: '00:02:46', timestamp_sec: 166, score: 0.70, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['bus'] },
  { id: '10', video_id: 'L03_V010', frame_id: 1402, timestamp: '00:02:51', timestamp_sec: 171, score: 0.66, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['traffic'] },
  { id: '11', video_id: 'L03_V011', frame_id: 1420, timestamp: '00:02:58', timestamp_sec: 178, score: 0.63, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['airplane'] },
  { id: '12', video_id: 'L03_V012', frame_id: 1438, timestamp: '00:03:04', timestamp_sec: 184, score: 0.60, thumbnail_url: '', video_url: 'https://www.w3schools.com/html/mov_bbb.mp4', objects: ['person', 'umbrella'] },
];

interface SpatialFilter {
  enabled: boolean;
  logic: 'AND' | 'OR';
  selectedObjects: string[];
  selectedColors: string[];
  maxObjectsText: string;
}

interface SearchStore {
  query: string;
  scene2Query: string;
  taskMode: TaskMode;
  modelEngine: ModelEngine;
  simReorder: boolean;
  advancedMode: boolean;
  isLoading: boolean;
  results: KeyframeResult[];
  selectedResultId: string;
  isVideoModalOpen: boolean;

  scene1Spatial: SpatialFilter;
  scene2Spatial: SpatialFilter;
  visualQuery: { imageUrl: string };

  setQuery: (q: string) => void;
  setScene2Query: (q: string) => void;
  setTaskMode: (m: TaskMode) => void;
  setModelEngine: (e: ModelEngine) => void;
  setSimReorder: (s: boolean) => void;
  setAdvancedMode: (a: boolean) => void;
  selectResult: (id: string) => void;
  setVideoModalOpen: (open: boolean) => void;
  executeSearch: () => void;
  runMockSearch: () => void;

  updateSpatialState: (scene: 'scene1' | 'scene2', patch: Partial<SpatialFilter>) => void;
  toggleObjectInSpatial: (scene: 'scene1' | 'scene2', obj: string) => void;
  toggleColorInSpatial: (scene: 'scene1' | 'scene2', color: string) => void;
  clearSpatialState: (scene: 'scene1' | 'scene2') => void;
  updateVisualQuery: (patch: { imageUrl: string }) => void;
}

export const useSearchStore = create<SearchStore>((set, get) => ({
  query: '',
  scene2Query: '',
  taskMode: 'KIS',
  modelEngine: 'use all',
  simReorder: true,
  advancedMode: false,
  isLoading: false,
  results: initialMockResults,
  selectedResultId: '1',
  isVideoModalOpen: false,

  scene1Spatial: {
    enabled: true,
    logic: 'AND',
    selectedObjects: [],
    selectedColors: [],
    maxObjectsText: '',
  },  
  scene2Spatial: {
    enabled: true,
    logic: 'AND',
    selectedObjects: [],
    selectedColors: [],
    maxObjectsText: '',
  },
  visualQuery: { imageUrl: '' },

  setQuery: (query) => set({ query }),
  setScene2Query: (scene2Query) => set({ scene2Query }),
  setTaskMode: (taskMode) => set({ taskMode }),
  setModelEngine: (modelEngine) => set({ modelEngine }),
  setSimReorder: (simReorder) => set({ simReorder }),
  setAdvancedMode: (advancedMode) => set({ advancedMode }),
  selectResult: (id) => set({ selectedResultId: id }),
  setVideoModalOpen: (open) => set({ isVideoModalOpen: open }),

  executeSearch: async () => {
    const state = get();
    const query = state.query.trim();
    if (!query) return;
    set({ isLoading: true });
    try {
      const trakeEvents =
        state.taskMode === 'TRAKE'
          ? [query, state.scene2Query.trim()].filter(Boolean)
          : undefined;
      const requestTaskMode =
        state.taskMode === 'TRAKE' && (!trakeEvents || trakeEvents.length < 2)
          ? 'KIS'
          : state.taskMode;
      const results = await searchApi.search({
        task_mode: requestTaskMode,
        query,
        model_engine: state.modelEngine,
        sim_reorder: state.simReorder,
        spatial_filter: {
          enabled: state.scene1Spatial.enabled,
          logic: state.scene1Spatial.logic,
          objects: state.scene1Spatial.selectedObjects,
          colors: state.scene1Spatial.selectedColors,
        },
        visual_query: state.visualQuery.imageUrl
          ? { image_url: state.visualQuery.imageUrl }
          : undefined,
        trake_events: requestTaskMode === 'TRAKE' ? trakeEvents : undefined,
        gap_phrases: requestTaskMode === 'TRAKE' && trakeEvents && trakeEvents.length > 1
          ? Array(trakeEvents.length - 1).fill('')
          : undefined,
        top_k: 100,
      });
      set({
        results,
        selectedResultId: results[0]?.id || '',
        isLoading: false,
      });
    } catch (error) {
      console.error('Search failed:', error);
      set({ isLoading: false });
      const detail =
        typeof error === 'object' &&
        error !== null &&
        'response' in error &&
        typeof error.response === 'object' &&
        error.response !== null &&
        'data' in error.response &&
        typeof error.response.data === 'object' &&
        error.response.data !== null &&
        'detail' in error.response.data
          ? String(error.response.data.detail)
          : 'Kiểm tra backend /health và console.';
      alert(`Không gọi được backend search. ${detail}`);
    }
  },

  runMockSearch: () => {
    set({ query: 'motorcycle on road', isLoading: true });
    setTimeout(() => {
      set({ results: initialMockResults, isLoading: false });
    }, 200);
  },

  updateSpatialState: (scene, patch) =>
    set((state) => ({
      [scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial']: {
        ...state[scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial'],
        ...patch,
      },
    })),

  toggleObjectInSpatial: (scene, obj) =>
    set((state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      const cur = state[key].selectedObjects;
      const next = cur.includes(obj) ? cur.filter((x) => x !== obj) : [...cur, obj];
      return { [key]: { ...state[key], selectedObjects: next } };
    }),

  toggleColorInSpatial: (scene, color) =>
    set((state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      const cur = state[key].selectedColors;
      const next = cur.includes(color) ? cur.filter((x) => x !== color) : [...cur, color];
      return { [key]: { ...state[key], selectedColors: next } };
    }),

  clearSpatialState: (scene) =>
    set((state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      return {
        [key]: {
          ...state[key],
          selectedObjects: [],
          selectedColors: [],
          maxObjectsText: '',
        },
      };
    }),

  updateVisualQuery: (patch) =>
    set((state) => ({ visualQuery: { ...state.visualQuery, ...patch } })),
}));
