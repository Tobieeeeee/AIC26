import type { KeyframeResult } from '../src/types/competition';

export function makeResult(overrides: Partial<KeyframeResult> = {}): KeyframeResult {
  return {
    keyframe_id: 'L01_V001_S0001_K01',
    video_id: 'L01_V001',
    frame_id: 1200,
    shot_id: 'L01_V001_S0001',
    timestamp: '00:00:40',
    timestamp_sec: 40,
    thumbnail_url: '/keyframes/L01_V001_S0001_K01.jpg',
    video_url: '/videos/L01_V001.mp4',
    score: 0.9,
    objects: ['person'],
    rank: 1,
    source: 'semantic',
    ocr_text: 'sample text',
    asr_text: 'sample speech',
    ...overrides,
  };
}

export function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });

  return { promise, resolve, reject };
}
