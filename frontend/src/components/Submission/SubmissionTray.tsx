import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from '@dnd-kit/core';
import {
  SortableContext,
  sortableKeyboardCoordinates,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { Download, ListChecks, Trash2 } from 'lucide-react';
import { useSubmissionStore } from '../../stores/useSubmissionStore';
import { SortableItem } from './SortableItem';

export function SubmissionTray() {
  const { items, removeItem, reorderItems, clearAll, maxItems } = useSubmissionStore();

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      reorderItems(active.id.toString(), over.id.toString());
    }
  };

  const handleExportCSV = () => {
    if (items.length === 0) {
      alert('Khay nộp bài đang trống!');
      return;
    }
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      items.map((item) => `${item.video_id},${item.frame_id}`).join('\n');

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `submission_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section className="submission-tray-container" aria-label="Submission tray">
      <div className="tray-top-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <ListChecks size={35} aria-hidden="true" />
          <h2 className="tray-main-title">Submission Tray</h2>
        </div>
        <div className="tray-btn-group">
          <button
            type="button"
            className="action-btn export-btn"
            onClick={handleExportCSV}
            disabled={items.length === 0}
            title="Xuất file CSV nộp bài"
          >
            <Download size={25} />
            Export
          </button>
          <button
            type="button"
            className="action-btn clear-btn"
            onClick={clearAll}
            disabled={items.length === 0}
            title="Xóa danh sách"
          >
            <Trash2 size={25} />
            Clear
          </button>
        </div>
      </div>

      <p className="tray-count-indicator">
        Selected {items.length} / {maxItems} (Kéo thả để đổi thứ tự)
      </p>

      <div className="tray-content-area">
        {items.length === 0 ? (
          <p className="empty-tray-notice">Khay nộp bài đang trống.</p>
        ) : (
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={items.map((i) => i.id)}
              strategy={verticalListSortingStrategy}
            >
              <div className="tray-pills-wrap-dnd">
                {items.map((item, index) => (
                  <SortableItem
                    key={item.id}
                    item={item}
                    index={index}
                    onRemove={removeItem}
                  />
                ))}
              </div>
            </SortableContext>
          </DndContext>
        )}
      </div>
    </section>
  );
}