import { afterEach, describe, expect, it, vi } from 'vitest';
import { mockResults } from '../src/data/mockResults';

describe('mock search safety', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it('keeps fixture mode disabled by default and does not load fixtures', async () => {
    vi.stubEnv('VITE_USE_MOCK_SEARCH', 'false');
    const { isMockSearchEnabled, useSearchStore } = await import(
      '../src/stores/useSearchStore'
    );

    expect(isMockSearchEnabled).toBe(false);
    expect(useSearchStore.getState().results).toEqual([]);

    useSearchStore.getState().runMockSearch();

    expect(useSearchStore.getState()).toMatchObject({
      status: 'error',
      results: [],
      selectedResultId: null,
      error: 'Fixture mode is disabled. Set VITE_USE_MOCK_SEARCH=true for local demos.',
    });
  });

  it('loads fixtures only when the explicit environment flag is enabled', async () => {
    vi.stubEnv('VITE_USE_MOCK_SEARCH', 'true');
    const { isMockSearchEnabled, useSearchStore } = await import(
      '../src/stores/useSearchStore'
    );

    expect(isMockSearchEnabled).toBe(true);
    expect(useSearchStore.getState().results).toEqual([]);

    useSearchStore.getState().runMockSearch();

    expect(useSearchStore.getState()).toMatchObject({
      status: 'success',
      error: null,
      results: mockResults,
      selectedResultId: mockResults[0].keyframe_id,
    });
  });

  it('defines fixture data with unique canonical keyframe identities', () => {
    const keyframeIds = mockResults.map((result) => result.keyframe_id);

    expect(new Set(keyframeIds).size).toBe(keyframeIds.length);
    expect(
      mockResults.every(
        (result) =>
          result.keyframe_id.length > 0 &&
          result.video_id.length > 0 &&
          Number.isFinite(result.frame_id) &&
          !/^\d+$/.test(result.keyframe_id) &&
          !/^kf-\d+$/i.test(result.keyframe_id),
      ),
    ).toBe(true);
  });
});
