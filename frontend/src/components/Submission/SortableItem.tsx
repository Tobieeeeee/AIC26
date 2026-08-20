import type { FC } from 'react';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { GripVertical, X } from 'lucide-react';
import type { SubmissionItem } from '../../types/competition';

interface SortableItemProps {
  item: SubmissionItem;
  index: number;
  onRemove: (id: string) => void;
}

export const SortableItem: FC<SortableItemProps> = ({ item, index, onRemove }) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: item.keyframe_id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  return (
    <div ref={setNodeRef} style={style} className="submission-item-row">
      <button
        type="button"
        {...attributes}
        {...listeners}
        className="drag-handle-btn"
        title="Kéo để đổi thứ tự ưu tiên"
      >
        <GripVertical size={14} />
      </button>

      <span className="sub-index">#{index + 1}</span>
      <span className="sub-vid"><strong>{item.video_id}</strong> - F{item.frame_id}</span>

      <button
        type="button"
        className="sub-remove-btn"
        onClick={() => onRemove(item.keyframe_id)}
        title="Xóa"
      >
        <X size={14} />
      </button>
    </div>
  );
};
