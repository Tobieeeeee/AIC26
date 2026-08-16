from importlib.resources import as_file, files

import torch
from transnetv2_pytorch import TransNetV2

from ..models import Shot, VideoInfo


class TransNetV2Detector:
    name = "transnetv2"

    def __init__(
        self,
        threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.threshold = threshold
        self.device = device

        # Model được tạo một lần ở đây, không tạo lại cho từng video.
        self.model = TransNetV2(device=device)

        weights_resource = files(
            "transnetv2_pytorch"
        ).joinpath(
            "transnetv2-pytorch-weights.pth"
        )

        with as_file(weights_resource) as weights_path:
            state_dict = torch.load(
                weights_path,
                map_location=self.model.device,
                weights_only=True,
            )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def detect(self, video: VideoInfo) -> list[Shot]:
        with torch.no_grad():
            scenes = self.model.detect_scenes(
                str(video.path),
                threshold=self.threshold,
            )

        last_video_frame = video.frame_count - 1

        if not scenes:
            return [
                Shot(
                    video_id=video.video_id,
                    method=self.name,
                    shot_id=1,
                    start_frame=0,
                    end_frame=last_video_frame,
                    fps=video.fps,
                )
            ]

        shots: list[Shot] = []

        for shot_id, scene in enumerate(scenes, start=1):
            start_frame = int(scene["start_frame"])
            end_frame = int(scene["end_frame"])

            start_frame = max(
                0,
                min(start_frame, last_video_frame),
            )

            end_frame = max(
                start_frame,
                min(end_frame, last_video_frame),
            )

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

        return shots
