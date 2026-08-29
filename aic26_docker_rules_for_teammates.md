# AIC26 — Docker Rules for Teammates

## Mục đích

Tài liệu này quy định cách sử dụng Docker trong repo AIC26 để đảm bảo:

- code chạy lại được trên máy khác;
- không phụ thuộc cấu hình cá nhân;
- không phá môi trường `conda mvr` của host;
- CPU/GPU chỉ là khác biệt runtime, không tạo hai pipeline khác nhau;
- model, checkpoint và artifacts được quản lý sạch;
- teammate có thể phát triển độc lập nhưng sau khi merge/pull về máy khác vẫn chạy được.

---

## 1. Dockerfile là source of truth của môi trường

Mọi dependency cần để chạy một component phải được khai báo trong:

```text
Dockerfile
requirements*.txt
pyproject.toml
config runtime tương ứng
```

Không chấp nhận kiểu:

```text
"Máy tôi đã cài sẵn package này nên chạy được."
```

Nếu một package là requirement thực tế của component, nó phải được mô tả trong repo.

---

## 2. Không sửa `conda mvr` chỉ để phục vụ riêng một model

Host environment là môi trường chính của project.

Ví dụ:

```text
Windows host
└── conda mvr
    └── Python 3.11.x
```

Nếu một model cần stack riêng, hãy đưa model đó vào Docker.

Ví dụ:

```text
docker/ocr/
docker/asr/
docker/milvus/
```

Không downgrade Python/Torch/CUDA trong host chỉ để chạy một component riêng.

---

## 3. Không hardcode path của máy cá nhân

Cấm các path như:

```text
D:\UserA\Desktop\data
C:\Users\ABC\Downloads\model
/home/teammate/data
```

Path phải truyền qua CLI, config hoặc Docker volume.

Ví dụ:

```text
--video-root /data/videos
--mapping /data/keyframe_mapping.csv
--model-dir /models
--output-dir /output
```

---

## 4. CPU và GPU chỉ là runtime khác nhau

Không được tạo hai pipeline logic khác nhau cho CPU và GPU.

Ví dụ với ASR:

```text
CPU teammate:
--device cpu

GPU machine:
--device cuda
```

Cùng một worker, cùng business logic, cùng output schema.

---

## 5. ASR bắt buộc có CPU-compatible path

Với Task04C ASR:

- CPU path là bắt buộc để development và smoke test;
- GPU là acceleration;
- teammate không có GPU vẫn phải phát triển được pipeline.

Không được biến GPU thành requirement bắt buộc của ASR nếu model có thể chạy CPU.

---

## 6. CPU/GPU phải sinh cùng output contract

Ví dụ canonical ASR:

```text
<video_id>/<keyframe_id>.json
```

Schema phải thống nhất:

```json
{
  "video_id": "L21_V010",
  "keyframe_id": "000123",
  "frame_idx": 3690,
  "timestamp": 123.0,
  "shot_id": "shot_42",
  "text": "...",
  "segments": []
}
```

Device không được làm thay đổi schema.

---

## 7. Pin version các dependency quan trọng

Không để các package quan trọng tự động lấy `latest`.

Ví dụ không nên để:

```text
torch
torchvision
faster-whisper
transformers
opencv-python
pytorch-lightning
```

mà không có version strategy.

Base Docker image cũng phải dùng tag cụ thể.

---

## 8. Không sửa container thủ công rồi coi đó là fix

Các lệnh kiểu:

```bash
docker exec ...
pip install ...
apt install ...
```

có thể dùng để debug.

Nhưng fix cuối cùng phải được đưa trở lại:

```text
Dockerfile
requirements
source code
config
scripts
```

Sau đó rebuild lại image.

---

## 9. Clean clone phải build được

Acceptance không phải:

```text
"Container trên máy teammate đang chạy được."
```

Acceptance phải gần với:

```text
git clone
→ chuẩn bị model/input theo hướng dẫn
→ docker build
→ docker run
→ smoke test PASS
```

---

## 10. Không commit Docker image vào Git

Repo chỉ nên chứa:

```text
Dockerfile
compose.yaml nếu cần
requirements
scripts build/run
source code
tests
README
```

Không commit:

```text
Docker image .tar
Docker layer cache
Docker Desktop data
WSL VHDX
CUDA image
```

---

## 11. Không commit model weights và production artifacts

Không commit các file lớn như:

```text
*.pth
*.ckpt
*.bin
*.onnx
model cache
full OCR output
full ASR output
runtime frames
full intermediate JSON corpus
```

Ngoại lệ: fixture rất nhỏ, deterministic, được chủ động dùng cho unit/integration test.

---

## 12. Docker volume phải phân rõ input/output

Khuyến nghị:

```text
/data
    input read-only

/models
    checkpoints / model cache

/output
    generated artifacts
```

Nếu input không cần ghi, mount read-only.

Container không được ghi bừa vào source tree.

---

## 13. Không có automatic mock/dummy fallback trong production

Nếu real model/checkpoint không load được:

```text
FAIL
```

Không được:

```text
real model lỗi
→ tự chuyển mock
→ pipeline vẫn PASS
```

Mock chỉ được phép trong test mode explicit.

---

## 14. Component dependency conflict thì tách image

Không cố nhồi mọi thứ vào một image lớn.

Ví dụ:

```text
mvr-ocr-deepsolo
mvr-ocr-parseq
mvr-asr
```

Các component giao tiếp qua artifact/API contract.

---

## 15. Docker không được chứa business logic chính

Các logic như:

```text
ASR segment → keyframe alignment
temporal window
shot boundary
canonical JSON generation
validation
```

phải nằm trong source code có test.

Docker chỉ chịu trách nhiệm runtime/environment.

---

## 16. Phải có command build/run chuẩn trong repo

Teammate phải để lại command reproducible.

Ví dụ:

```powershell
docker build -f docker/asr/Dockerfile.cpu -t mvr-asr:cpu .
```

hoặc script:

```text
scripts/build_asr.ps1
scripts/run_asr.ps1
```

Người khác không được phải đoán cách build.

---

## 17. Phải có real smoke test

Sau build phải chứng minh tối thiểu:

```text
container startup PASS
import model PASS
checkpoint/model load PASS
one real input PASS
output schema PASS
```

Nếu là GPU runtime:

```text
GPU visible PASS
CUDA inference PASS
```

---

## 18. `docker build` PASS chưa có nghĩa model PASS

Build thành công chỉ chứng minh image tạo được.

Acceptance cần:

```text
docker build
→ container startup
→ model load
→ real inference
→ output validation
```

---

## 19. Manifest phải ghi runtime thực tế

Production artifact nên lưu các thông tin như:

```text
Docker image tag
Docker image ID
Python version
model name/version
source repo
source commit
checkpoint path
checkpoint SHA256
device
compute type
runtime configuration
```

Không để:

```text
TBD
TODO
dummy
mock
```

trong accepted production manifest.

---

## 20. Không phụ thuộc image có sẵn trên máy teammate

Khi push code phải giả định người khác chưa có image.

Điều cần reproducible là:

```text
docker build từ source trong repo
```

không phải:

```text
dùng đúng container đang nằm trong Docker Desktop của teammate
```

---

## 21. Không tự ý đổi model/runtime chỉ để "cho chạy được"

Nếu gặp:

```text
dependency conflict
CUDA incompatibility
checkpoint load failure
model runtime failure
```

phải:

1. tìm root cause;
2. report chính xác;
3. sửa trong phạm vi architecture được giao.

Không âm thầm đổi model khác.

---

## 22. Thay đổi Docker không được phá component khác

Task ASR không được vô tình chỉnh semantics của:

```text
OCR
CLIP
object extraction
ingestion
keyframe extraction
keyframe_mapping.csv
retrieval
```

trừ khi task có yêu cầu rõ ràng.

---

## 23. Trước khi bàn giao phải chạy Git safety check

Chạy:

```powershell
git status --short
git diff --check
git diff --cached --name-only
```

Đảm bảo không stage nhầm:

```text
*.pth
*.ckpt
large model files
runtime artifacts
full generated outputs
real frame dumps
temporary Docker files
```

---

## 24. Tiêu chí bàn giao là "máy người khác chạy được"

Không dùng tiêu chí:

> Máy em chạy được.

Tiêu chí phải là:

> Người khác pull branch, build image từ Dockerfile trong repo, cung cấp input/model theo hướng dẫn và chạy được cùng pipeline.

---

# Quy định riêng cho Task04C ASR

## 25. ASR phải giữ raw transcript và canonical keyframe output

Pipeline:

```text
video
→ audio extraction
→ ASR timestamped segments
→ align với canonical keyframe mapping
→ one JSON per keyframe
```

Raw transcript vẫn phải được giữ để provenance/debug.

---

## 26. Canonical ASR phải align theo keyframe

Output dùng cho downstream indexing nên có identity:

```text
(video_id, keyframe_id)
```

Khuyến nghị:

```text
artifacts/features_<dataset>/asr_raw/<video_id>.json
artifacts/features_<dataset>/asr/<video_id>/<keyframe_id>.json
```

---

## 27. Không hardcode việc mỗi keyframe luôn có speech

Keyframe không có speech vẫn phải có canonical output hợp lệ:

```json
{
  "video_id": "L21_V010",
  "keyframe_id": "000124",
  "frame_idx": 3720,
  "timestamp": 124.0,
  "shot_id": "shot_42",
  "text": "",
  "segments": []
}
```

Nếu canonical mapping có N keyframes thì ASR canonical phải hướng tới N JSON files.

---

## 28. ASR alignment phải dựa trên mapping hiện tại

Không tự tạo keyframe mới.

Không derive identity từ filename nếu mapping đã có.

Dùng canonical `keyframe_mapping.csv` làm source of truth cho:

```text
video_id
keyframe_id
frame_idx
timestamp
shot_id
```

---

## 29. CPU teammate và GPU máy chính phải dùng cùng worker

Ví dụ:

```text
src/mvr/enrichment/asr/
```

CPU:

```text
--device cpu
--compute-type int8
```

GPU:

```text
--device cuda
--compute-type float16
```

Không fork business logic thành hai codebase.

---

## 30. Teammate không GPU vẫn phải hoàn thành được phần lớn Task04C

Teammate có thể chịu trách nhiệm:

```text
Docker CPU
ffmpeg
ASR worker
raw transcript schema
keyframe alignment
canonical ASR JSON
manifest
unit tests
subset E2E
validator
```

Máy GPU có thể đảm nhiệm:

```text
final GPU verification
full dataset production inference
```

---

# Checklist trước khi teammate push branch

```text
[ ] Không hardcode path cá nhân
[ ] Dockerfile nằm trong repo
[ ] Dependency versions có strategy rõ
[ ] CPU path chạy được
[ ] GPU path không thay đổi schema
[ ] Không có production mock fallback
[ ] Real smoke test PASS
[ ] Model/checkpoint không nằm trong Git
[ ] Generated artifacts không nằm trong Git
[ ] Build/run command được ghi lại
[ ] Manifest có runtime metadata
[ ] git diff --check PASS
[ ] git status đã audit
[ ] Người khác có thể rebuild image từ repo
```

---

# Nguyên tắc cuối

Docker được dùng để **đóng gói runtime**, không phải để che giấu một môi trường chỉ chạy được trên một máy.

Mục tiêu là:

```text
reproducible code
+ reproducible Docker environment
+ stable input/output contract
```

để sau khi merge/pull, component có thể được build lại và chạy trên máy khác mà không phải sửa thủ công.
