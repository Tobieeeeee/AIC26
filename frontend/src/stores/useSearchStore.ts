import { create } from 'zustand';
import { searchApi, type SearchPayload } from '../api/searchApi';
import { mockResults } from '../data/mockResults';
import type { KeyframeResult, RetrievalMode, TaskMode } from '../types/competition';

export const isMockSearchEnabled = import.meta.env.VITE_USE_MOCK_SEARCH === 'true';

type SearchStatus = 'idle' | 'loading' | 'success' | 'success-empty' | 'error';

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
  retrievalMode: RetrievalMode;
  simReorder: boolean;
  advancedMode: boolean;
  status: SearchStatus;
  isLoading: boolean;
  error: string | null;
  results: KeyframeResult[];
  selectedResultId: string | null;
  isVideoModalOpen: boolean;

  scene1Spatial: SpatialFilter;
  scene2Spatial: SpatialFilter;
  visualQuery: { imageUrl: string };

  setQuery: (query: string) => void;
  setScene2Query: (query: string) => void;
  setTaskMode: (taskMode: TaskMode) => void;
  setRetrievalMode: (retrievalMode: RetrievalMode) => void;
  setSimReorder: (simReorder: boolean) => void;
  setAdvancedMode: (advancedMode: boolean) => void;
  selectResult: (keyframeId: string) => void;
  setVideoModalOpen: (open: boolean) => void;
  executeSearch: () => Promise<void>;
  runMockSearch: () => void;

  updateSpatialState: (scene: 'scene1' | 'scene2', patch: Partial<SpatialFilter>) => void;
  toggleObjectInSpatial: (scene: 'scene1' | 'scene2', object: string) => void;
  toggleColorInSpatial: (scene: 'scene1' | 'scene2', color: string) => void;
  clearSpatialState: (scene: 'scene1' | 'scene2') => void;
  updateVisualQuery: (patch: { imageUrl: string }) => void;
}

let latestRequestId = 0;

const emptySpatialFilter = (): SpatialFilter => ({
  enabled: true,
  logic: 'AND',
  selectedObjects: [],
  selectedColors: [],
  maxObjectsText: '',
});

export const useSearchStore = create<SearchStore>((set, get) => ({
  query: '',
  scene2Query: '',
  taskMode: 'kis',
  retrievalMode: 'hybrid',
  simReorder: true,
  advancedMode: false,
  status: 'idle',
  isLoading: false,
  error: null,
  results: [],
  selectedResultId: null,
  isVideoModalOpen: false,

  scene1Spatial: emptySpatialFilter(),
  scene2Spatial: emptySpatialFilter(),
  visualQuery: { imageUrl: '' },

  setQuery: (query) => set({ query }),
  setScene2Query: (scene2Query) => set({ scene2Query }),
  setTaskMode: (taskMode) => set({ taskMode }),
  setRetrievalMode: (retrievalMode) => set({ retrievalMode }),
  setSimReorder: (simReorder) => set({ simReorder }),
  setAdvancedMode: (advancedMode) => set({ advancedMode }),
  selectResult: (selectedResultId) => set({ selectedResultId }),
  setVideoModalOpen: (isVideoModalOpen) => set({ isVideoModalOpen }),

  executeSearch: async () => {
    const state = get();
    const query = state.query.trim();
    const requestId = ++latestRequestId;

    if (!query && !state.visualQuery.imageUrl.trim()) {
      set({
        status: 'error',
        isLoading: false,
        error: 'Enter a text query or provide an image before searching.',
        results: [],
        selectedResultId: null,
        isVideoModalOpen: false,
      });
      return;
    }

    const payload: SearchPayload = {
      task_mode: state.taskMode,
      query,
      retrieval_mode: state.retrievalMode,
      sim_reorder: state.simReorder,
      top_k: 100,
      spatial_filter: {
        enabled: state.scene1Spatial.enabled,
        logic: state.scene1Spatial.logic,
        objects: state.scene1Spatial.selectedObjects,
        colors: state.scene1Spatial.selectedColors,
        max_objects_text: state.scene1Spatial.maxObjectsText || undefined,
      },
      visual_query: state.visualQuery.imageUrl
        ? { image_url: state.visualQuery.imageUrl }
        : undefined,
    };

    set({
      status: 'loading',
      isLoading: true,
      error: null,
      results: [],
      selectedResultId: null,
      isVideoModalOpen: false,
    });

    try {
      const results = await searchApi.search(payload);
      if (requestId !== latestRequestId) return;
      set({
        status: results.length > 0 ? 'success' : 'success-empty',
        isLoading: false,
        error: null,
        results,
        selectedResultId: results[0]?.keyframe_id ?? null,
      });
    } catch (error) {
      if (requestId !== latestRequestId) return;
      set({
        status: 'error',
        isLoading: false,
        error: error instanceof Error ? error.message : 'Search failed.',
        results: [],
        selectedResultId: null,
        isVideoModalOpen: false,
      });
    }
  },

  runMockSearch: () => {
    if (!isMockSearchEnabled) {
      set({
        status: 'error',
        error: 'Fixture mode is disabled. Set VITE_USE_MOCK_SEARCH=true for local demos.',
        results: [],
        selectedResultId: null,
      });
      return;
    }
    set({
      query: get().query || 'fixture demo search',
      status: 'success',
      isLoading: false,
      error: null,
      results: mockResults,
      selectedResultId: mockResults[0]?.keyframe_id ?? null,
      isVideoModalOpen: false,
    });
  },

  updateSpatialState: (scene, patch) =>
    set((state) => ({
      [scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial']: {
        ...state[scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial'],
        ...patch,
      },
    })),

  toggleObjectInSpatial: (scene, object) =>
    set((state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      const current = state[key].selectedObjects;
      const selectedObjects = current.includes(object)
        ? current.filter((value) => value !== object)
        : [...current, object];
      return { [key]: { ...state[key], selectedObjects } };
    }),

  toggleColorInSpatial: (scene, color) =>
    set((state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      const current = state[key].selectedColors;
      const selectedColors = current.includes(color)
        ? current.filter((value) => value !== color)
        : [...current, color];
      return { [key]: { ...state[key], selectedColors } };
    }),

  clearSpatialState: (scene) =>
    set((_state) => {
      const key = scene === 'scene1' ? 'scene1Spatial' : 'scene2Spatial';
      return { [key]: emptySpatialFilter() };
    }),

  updateVisualQuery: (patch) =>
    set((state) => ({ visualQuery: { ...state.visualQuery, ...patch } })),
}));
