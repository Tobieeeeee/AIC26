import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { searchApi, type SearchPayload } from '../src/api/searchApi';
import { useSearchStore } from '../src/stores/useSearchStore';
import { deferred, makeResult } from './testData';

const initialState = useSearchStore.getInitialState();

describe('useSearchStore', () => {
  beforeEach(() => {
    useSearchStore.setState(initialState, true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('starts idle with no result, selection, or error state', () => {
    const state = useSearchStore.getState();

    expect(state.status).toBe('idle');
    expect(state.isLoading).toBe(false);
    expect(state.results).toEqual([]);
    expect(state.selectedResultId).toBeNull();
    expect(state.error).toBeNull();
  });

  it('moves from loading to success and replaces prior state', async () => {
    const request = deferred<ReturnType<typeof makeResult>[]>();
    const result = makeResult();
    vi.spyOn(searchApi, 'search').mockReturnValue(request.promise);
    useSearchStore.setState({
      query: '  a motorcycle  ',
      results: [makeResult({ keyframe_id: 'OLD', video_id: 'OLD' })],
      selectedResultId: 'OLD',
      error: 'old error',
      isVideoModalOpen: true,
    });

    const search = useSearchStore.getState().executeSearch();

    expect(useSearchStore.getState()).toMatchObject({
      status: 'loading',
      isLoading: true,
      error: null,
      results: [],
      selectedResultId: null,
      isVideoModalOpen: false,
    });

    request.resolve([result]);
    await search;

    expect(useSearchStore.getState()).toMatchObject({
      status: 'success',
      isLoading: false,
      error: null,
      results: [result],
      selectedResultId: result.keyframe_id,
    });
  });

  it('moves from loading to success-empty for an empty API result', async () => {
    vi.spyOn(searchApi, 'search').mockResolvedValue([]);
    useSearchStore.setState({ query: 'nothing matches' });

    await useSearchStore.getState().executeSearch();

    expect(useSearchStore.getState()).toMatchObject({
      status: 'success-empty',
      isLoading: false,
      error: null,
      results: [],
      selectedResultId: null,
    });
  });

  it('surfaces API errors and never falls back to fixtures', async () => {
    vi.spyOn(searchApi, 'search').mockRejectedValue(new Error('backend unavailable'));
    useSearchStore.setState({ query: 'find this scene' });

    await useSearchStore.getState().executeSearch();

    expect(useSearchStore.getState()).toMatchObject({
      status: 'error',
      isLoading: false,
      error: 'backend unavailable',
      results: [],
      selectedResultId: null,
      isVideoModalOpen: false,
    });
  });

  it('rejects a search without text or an image before calling the API', async () => {
    const search = vi.spyOn(searchApi, 'search');

    await useSearchStore.getState().executeSearch();

    expect(search).not.toHaveBeenCalled();
    expect(useSearchStore.getState()).toMatchObject({
      status: 'error',
      isLoading: false,
      error: 'Enter a text query or provide an image before searching.',
      results: [],
      selectedResultId: null,
      isVideoModalOpen: false,
    });
  });

  it('builds the current normalized payload at the API boundary', async () => {
    const search = vi.spyOn(searchApi, 'search').mockResolvedValue([]);
    useSearchStore.setState({
      query: '  red car  ',
      taskMode: 'qa',
      retrievalMode: 'object',
      simReorder: false,
      scene1Spatial: {
        enabled: true,
        logic: 'OR',
        selectedObjects: ['car'],
        selectedColors: ['red'],
        maxObjectsText: '3',
      },
      visualQuery: { imageUrl: 'https://example.test/query.jpg' },
    });

    await useSearchStore.getState().executeSearch();

    expect(search).toHaveBeenCalledWith({
      task_mode: 'qa',
      query: 'red car',
      retrieval_mode: 'object',
      sim_reorder: false,
      top_k: 100,
      spatial_filter: {
        enabled: true,
        logic: 'OR',
        objects: ['car'],
        colors: ['red'],
        max_objects_text: '3',
      },
      visual_query: { image_url: 'https://example.test/query.jpg' },
    } satisfies SearchPayload);
  });

  it('allows an image-only search without synthesizing text', async () => {
    const search = vi.spyOn(searchApi, 'search').mockResolvedValue([]);
    useSearchStore.setState({
      visualQuery: { imageUrl: 'https://example.test/query.jpg' },
    });

    await useSearchStore.getState().executeSearch();

    expect(search).toHaveBeenCalledWith(
      expect.objectContaining({
        query: '',
        visual_query: { image_url: 'https://example.test/query.jpg' },
      }),
    );
  });

  it('keeps the newer response when an older request resolves last', async () => {
    const requestA = deferred<ReturnType<typeof makeResult>[]>();
    const requestB = deferred<ReturnType<typeof makeResult>[]>();
    const resultA = makeResult({ keyframe_id: 'A', video_id: 'VIDEO_A' });
    const resultB = makeResult({ keyframe_id: 'B', video_id: 'VIDEO_B' });
    vi.spyOn(searchApi, 'search')
      .mockReturnValueOnce(requestA.promise)
      .mockReturnValueOnce(requestB.promise);

    useSearchStore.setState({ query: 'request A' });
    const searchA = useSearchStore.getState().executeSearch();
    useSearchStore.setState({ query: 'request B' });
    const searchB = useSearchStore.getState().executeSearch();

    requestB.resolve([resultB]);
    await searchB;
    expect(useSearchStore.getState().results).toEqual([resultB]);

    requestA.resolve([resultA]);
    await searchA;
    expect(useSearchStore.getState()).toMatchObject({
      status: 'success',
      results: [resultB],
      selectedResultId: 'B',
    });
  });

  it('uses keyframe_id as selection identity independent of video/frame fields', () => {
    useSearchStore.getState().selectResult('CANONICAL_KEYFRAME_ID');

    expect(useSearchStore.getState().selectedResultId).toBe('CANONICAL_KEYFRAME_ID');
  });
});
