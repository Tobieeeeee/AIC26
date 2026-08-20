// src/utils/csvExporter.ts
export interface SubmissionRow {
    video_id: string;
    frame_ids: number[]; // Hỗ trợ 1 hoặc nhiều frame (TRAKE)
    answer?: string;     // Hỗ trợ Q&A
  }
  
  export function generateSubmissionCSV(rows: SubmissionRow[], taskMode: 'KIS' | 'QA' | 'TRAKE'): string {
    // Giới hạn tối đa 100 dòng theo thể lệ BTC
    const limitedRows = rows.slice(0, 100);
  
    return limitedRows
      .map((row) => {
        // 1. Loại bỏ phần đuôi mở rộng .mp4 nếu có
        const cleanVideoId = row.video_id.replace(/\.[^/.]+$/, '');
  
        if (taskMode === 'KIS') {
          // Format: <video_id>, <frame_id>
          return `${cleanVideoId}, ${row.frame_ids[0]}`;
        }
  
        if (taskMode === 'QA') {
          // Format: <video_id>, <frame_id>, "<answer>"
          // Escape dấu ngoặc kép bên trong answer nếu có
          const escapedAnswer = (row.answer || '').replace(/"/g, '""');
          return `${cleanVideoId}, ${row.frame_ids[0]}, "${escapedAnswer}"`;
        }
  
        if (taskMode === 'TRAKE') {
          // Format: <video_id>, <frame_id_1>, <frame_id_2>, ..., <frame_id_N>
          return `${cleanVideoId}, ${row.frame_ids.join(', ')}`;
        }
  
        return '';
      })
      .join('\r\n'); // Dùng CRLF hoặc LF theo chuẩn UTF-8
  }

  