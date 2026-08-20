import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { apiClient } from '../src/api/client';
import { searchApi, type SearchPayload } from '../src/api/searchApi';

const payload: SearchPayload = {
  task_mode: 'kis',
  query: 'a motorcycle',
  retrieval_mode: 'hybrid',
  sim_reorder: true,
  top_k: 100,
};

function mockResults(results: unknown[]) {
  return vi.spyOn(apiClient, 'post').mockResolvedValue({
    data: { results },
  } as never);
}

describe('searchApi', () => {
  beforeEach(() => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('normalizes a valid result and preserves supported optional fields', async () => {
    const post = mockResults([
      {
        keyframe_id: 'L01_V001_S0002_K01',
        video_id: 'L01_V001',
        frame_id: '42',
        shot_id: 'L01_V001_S0002',
        timestamp: 65.6,
        image_path: '/keyframes/frame-42.jpg',
        video_url: '/videos/L01_V001.mp4',
        score: 0.83,
        objects: ['person', 123, 'motorcycle'],
        rank: 2,
        source: 'hybrid',
        ocr_text: 'ROAD 66',
        asr_text: 'a spoken phrase',
      },
    ]);

    await expect(searchApi.search(payload)).resolves.toEqual([
      {
        keyframe_id: 'L01_V001_S0002_K01',
        video_id: 'L01_V001',
        frame_id: 42,
        shot_id: 'L01_V001_S0002',
        timestamp: '00:01:06',
        timestamp_sec: 65.6,
        thumbnail_url: '/keyframes/frame-42.jpg',
        video_url: '/videos/L01_V001.mp4',
        score: 0.83,
        objects: ['person', 'motorcycle'],
        rank: 2,
        source: 'hybrid',
        ocr_text: 'ROAD 66',
        asr_text: 'a spoken phrase',
      },
    ]);
    expect(post).toHaveBeenCalledWith('/search', payload);
  });

  it('preserves a formatted timestamp and accepts a separate numeric timestamp', async () => {
    mockResults([
      {
        keyframe_id: 'L01_V001_S0003_K01',
        video_id: 'L01_V001',
        frame_id: 84,
        timestamp: '00:02:15',
        timestamp_sec: 135,
        score: 0.5,
        thumbnail_url: '/thumbnail.jpg',
      },
    ]);

    const [result] = await searchApi.search(payload);

    expect(result.timestamp).toBe('00:02:15');
    expect(result.timestamp_sec).toBe(135);
    expect(result.thumbnail_url).toBe('/thumbnail.jpg');
  });

  it.each([
    ['keyframe_id', { video_id: 'L01_V001', frame_id: 1 }],
    ['video_id', { keyframe_id: 'L01_V001_S0001_K01', frame_id: 1 }],
    ['frame_id', { keyframe_id: 'L01_V001_S0001_K01', video_id: 'L01_V001' }],
    [
      'numeric frame_id',
      {
        keyframe_id: 'L01_V001_S0001_K01',
        video_id: 'L01_V001',
        frame_id: 'not-a-frame',
      },
    ],
  ])('rejects a result without a valid required %s', async (_field, result) => {
    mockResults([result]);

    await expect(searchApi.search(payload)).rejects.toThrow(
      'Search response is missing canonical keyframe identity.',
    );
  });

  it('rejects non-object result records', async () => {
    mockResults([null]);

    await expect(searchApi.search(payload)).rejects.toThrow(
      'Search response contained an invalid result.',
    );
  });

  it('rejects a response without a results array', async () => {
    vi.spyOn(apiClient, 'post').mockResolvedValue({ data: {} } as never);

    await expect(searchApi.search(payload)).rejects.toThrow(
      'Search response did not contain a results array.',
    );
  });

  it('never invents a sequential canonical ID', async () => {
    mockResults([{ video_id: 'L01_V001', frame_id: 7, timestamp: 1, score: 0.4 }]);

    await expect(searchApi.search(payload)).rejects.toThrow(
      'Search response is missing canonical keyframe identity.',
    );
  });
});
