from scenedetect import ContentDetector, detect

from ..models import Shot, VideoInfo


class PySceneDetector:
    name = "pyscenedetect"

    def __init__(self, threshold: float = 27.0) -> None:
        self.threshold = threshold

    def detect(self, video: VideoInfo) -> list[Shot]:
        scene_list = detect(
            str(video.path),
            ContentDetector(threshold=self.threshold),
            start_in_scene=True,
        )

        # Trường hợp detector không trả về shot nào.
        if not scene_list:
            return [
                Shot(
                    video_id=video.video_id,
                    method=self.name,
                    shot_id=1,
                    start_frame=0,
                    end_frame=video.frame_count - 1,
                    fps=video.fps,
                )
            ]

        shots: list[Shot] = []
        last_video_frame = video.frame_count - 1

        for shot_id, (scene_start, scene_end) in enumerate(scene_list, start=1):
            start_frame = int(scene_start.get_frames())

            # PySceneDetect trả scene_end theo dạng exclusive.
            # Lab yêu cầu end_frame inclusive nên phải trừ 1.
            end_frame = int(scene_end.get_frames()) - 1

            start_frame = max(0, start_frame)
            end_frame = min(last_video_frame, end_frame)

            if end_frame < start_frame:
                continue

            shots.append(
                Shot(
                    video_id=video.video_id,
                    method=self.name,
                    shot_id=shot_id,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    fps=video.fps,
                )
            )

        if not shots:
            shots.append(
                Shot(
                    video_id=video.video_id,
                    method=self.name,
                    shot_id=1,
                    start_frame=0,
                    end_frame=last_video_frame,
                    fps=video.fps,
                )
            )

        return shots
