import type { TaskMode } from '../types/competition';

export interface SubmissionRow {
  video_id: string;
  frame_ids: number[];
  answer?: string;
}

export function generateSubmissionCSV(rows: SubmissionRow[], taskMode: TaskMode): string {
  return rows
    .slice(0, 100)
    .map((row) => {
      const cleanVideoId = row.video_id.replace(/\.[^/.]+$/, '');

      if (taskMode === 'kis') {
        return `${cleanVideoId}, ${row.frame_ids[0] ?? ''}`;
      }

      if (taskMode === 'qa') {
        const escapedAnswer = (row.answer || '').replace(/"/g, '""');
        return `${cleanVideoId}, ${row.frame_ids[0] ?? ''}, "${escapedAnswer}"`;
      }

      if (taskMode === 'trake') {
        return `${cleanVideoId}, ${row.frame_ids.join(', ')}`;
      }

      return '';
    })
    .join('\r\n');
}
