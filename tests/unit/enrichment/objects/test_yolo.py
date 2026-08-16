import numpy as np


def test_yolov8n_preserves_model_confidence_and_per_frame_counts() -> None:
    from mvr.enrichment.objects.yolo import YoloObjectDetector

    calls = []

    class Boxes:
        cls = np.array([0, 2, 0], dtype=np.float32)

    class Result:
        names = {0: "person", 2: "car"}
        boxes = Boxes()

    class Model:
        def predict(self, image, conf, verbose):
            calls.append(("predict", image, conf, verbose))
            return [Result()]

    def factory(weights):
        calls.append(("factory", weights))
        return Model()

    image = object()
    detector = YoloObjectDetector(model_factory=factory)

    assert detector.detect(image) == {"person": 2, "car": 1}
    assert calls == [
        ("factory", "yolov8n.pt"),
        ("predict", image, 0.4, False),
    ]
