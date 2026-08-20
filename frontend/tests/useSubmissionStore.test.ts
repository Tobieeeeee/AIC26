import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { useSubmissionStore } from '../src/stores/useSubmissionStore';
import { makeResult } from './testData';

const initialState = useSubmissionStore.getInitialState();

describe('useSubmissionStore', () => {
  beforeEach(() => {
    useSubmissionStore.setState(initialState, true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('adds a valid canonical result', () => {
    const result = makeResult();

    useSubmissionStore.getState().addItem(result);

    expect(useSubmissionStore.getState().items).toEqual([
      {
        keyframe_id: result.keyframe_id,
        video_id: result.video_id,
        frame_id: result.frame_id,
        shot_id: result.shot_id,
        timestamp: result.timestamp,
        score: result.score,
        timestamp_sec: result.timestamp_sec,
        thumbnail_url: result.thumbnail_url,
        video_url: result.video_url,
        objects: result.objects,
      },
    ]);
  });

  it('prevents duplicates by keyframe_id even if other fields differ', () => {
    useSubmissionStore.getState().addItem(makeResult());
    useSubmissionStore.getState().addItem(
      makeResult({ video_id: 'DIFFERENT_VIDEO', frame_id: 9999 }),
    );

    expect(useSubmissionStore.getState().items).toHaveLength(1);
    expect(useSubmissionStore.getState().items[0].video_id).toBe('L01_V001');
  });

  it('keeps distinct keyframe_id values even when video_id and frame_id match', () => {
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_A' }));
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_B' }));

    expect(useSubmissionStore.getState().items.map((item) => item.keyframe_id)).toEqual([
      'KEYFRAME_A',
      'KEYFRAME_B',
    ]);
  });

  it('removes by canonical identity', () => {
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_A' }));
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_B' }));

    useSubmissionStore.getState().removeItem('KEYFRAME_A');

    expect(useSubmissionStore.getState().items.map((item) => item.keyframe_id)).toEqual([
      'KEYFRAME_B',
    ]);
  });

  it('reorders by canonical identity', () => {
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_A' }));
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_B' }));
    useSubmissionStore.getState().addItem(makeResult({ keyframe_id: 'KEYFRAME_C' }));

    useSubmissionStore.getState().reorderItems('KEYFRAME_C', 'KEYFRAME_A');

    expect(useSubmissionStore.getState().items.map((item) => item.keyframe_id)).toEqual([
      'KEYFRAME_C',
      'KEYFRAME_A',
      'KEYFRAME_B',
    ]);
  });

  it('enforces the store-owned 100-item limit', () => {
    const alert = vi.spyOn(window, 'alert').mockImplementation(() => undefined);
    for (let index = 0; index < 100; index += 1) {
      useSubmissionStore.getState().addItem(
        makeResult({ keyframe_id: `KEYFRAME_${index}`, frame_id: index }),
      );
    }

    useSubmissionStore.getState().addItem(
      makeResult({ keyframe_id: 'KEYFRAME_101', frame_id: 101 }),
    );

    expect(useSubmissionStore.getState().items).toHaveLength(100);
    expect(alert).toHaveBeenCalledOnce();
  });
});
