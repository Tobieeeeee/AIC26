import { describe, expect, it } from 'vitest';
import { generateSubmissionCSV } from '../src/utils/csvExporter';

describe('generateSubmissionCSV', () => {
  it('exports KIS rows in input order and strips video extensions', () => {
    expect(
      generateSubmissionCSV(
        [
          { video_id: 'L01_V002.mp4', frame_ids: [20] },
          { video_id: 'L01_V001', frame_ids: [10] },
        ],
        'kis',
      ),
    ).toBe('L01_V002, 20\r\nL01_V001, 10');
  });

  it('exports QA rows and escapes quotes in answers', () => {
    expect(
      generateSubmissionCSV(
        [
          {
            video_id: 'L01_V001.mp4',
            frame_ids: [25],
            answer: 'A "quoted", answer',
          },
        ],
        'qa',
      ),
    ).toBe('L01_V001, 25, "A ""quoted"", answer"');
  });

  it('exports all current TRAKE-like frame IDs in row order', () => {
    expect(
      generateSubmissionCSV(
        [{ video_id: 'L01_V001', frame_ids: [30, 10, 20] }],
        'trake',
      ),
    ).toBe('L01_V001, 30, 10, 20');
  });

  it('returns an empty string for empty input', () => {
    expect(generateSubmissionCSV([], 'kis')).toBe('');
  });

  it('keeps the current blank-frame behavior for missing KIS frame data', () => {
    expect(
      generateSubmissionCSV([{ video_id: 'L01_V001', frame_ids: [] }], 'kis'),
    ).toBe('L01_V001, ');
  });

  it('keeps the current blank-frame behavior for missing QA frame data', () => {
    expect(
      generateSubmissionCSV(
        [{ video_id: 'L01_V001', frame_ids: [], answer: 'unknown' }],
        'qa',
      ),
    ).toBe('L01_V001, , "unknown"');
  });

  it('limits export to the first 100 rows', () => {
    const rows = Array.from({ length: 101 }, (_, index) => ({
      video_id: `VIDEO_${index}`,
      frame_ids: [index],
    }));
    const lines = generateSubmissionCSV(rows, 'kis').split('\r\n');

    expect(lines).toHaveLength(100);
    expect(lines[0]).toBe('VIDEO_0, 0');
    expect(lines[99]).toBe('VIDEO_99, 99');
  });
});
