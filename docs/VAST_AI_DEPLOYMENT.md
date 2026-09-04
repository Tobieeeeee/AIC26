# Chay Okchokchok tren Vast.ai

Tai lieu nay huong dan chay day du frontend va backend Okchokchok tren mot
Vast.ai GPU instance, voi ket qua tim kiem that, keyframe va video summary.

## 1. Kien truc demo

```text
May ca nhan
  -> SSH tunnel
  -> Vast.ai instance
       |- Vite frontend       127.0.0.1:5173
       |- FastAPI backend     127.0.0.1:8000
       |- AIC dataset         /workspace/aic-data
       `- FAISS index         /workspace/aic-index
```

GitHub chi chua source code. Khong dua dataset, video, keyframe, model weight,
FAISS index hoac file `.env` len GitHub.

De search ra ket qua that, Vast instance phai truy cap duoc ca dataset va FAISS
index. Du lieu khong can nam tren may cua nguoi dung, nhung trong phien chay no
phai nam tren disk/volume cua Vast hoac duoc tai tu object storage vao do.

## 2. Chuan bi Vast instance

Tao mot instance tu PyTorch/CUDA template va bat SSH. Cau hinh goi y:

| Thanh phan | Toi thieu | Goi y |
|---|---:|---:|
| GPU | NVIDIA T4 16 GB | RTX 3090/4090 hoac A100 |
| RAM | 16 GB | 32 GB |
| Disk | Du cho subset demo | 100-200 GB cho corpus lon |
| Python | 3.10+ | 3.12 |
| Node.js | 20.19+ | 22 LTS |

Neu can giu dataset sau khi destroy instance, dung Vast Volume. Container disk
van con khi stop instance, nhung bi xoa khi destroy instance.

Tai lieu Vast:

- <https://docs.vast.ai/guides/instances/connect/ssh>
- <https://docs.vast.ai/guides/instances/storage/types>
- <https://docs.vast.ai/guides/instances/storage/volumes>

## 3. Ket noi SSH

Trong trang Instances cua Vast, bam nut SSH va sao chep lenh ket noi. Dang tong
quat:

```bash
ssh -p <SSH_PORT> root@<VAST_IP>
```

Kiem tra GPU va disk:

```bash
nvidia-smi
df -h
```

## 4. Clone dung branch

Tren Vast instance:

```bash
cd /workspace
git clone -b feat/backend-ui-api --single-branch \
  https://github.com/Tobieeeeee/AIC26.git
cd AIC26
git log --oneline -3
```

Commit UI/video summary can co trong log:

```text
f0ab0c8 feat: complete retrieval UI and video summary workflow
```

Neu repo da ton tai:

```bash
cd /workspace/AIC26
git fetch origin
git switch feat/backend-ui-api
git pull --ff-only origin feat/backend-ui-api
```

## 5. Cai backend

PyTorch template thuong da co CUDA va PyTorch. Tao virtual environment cho repo
va cho phep tai su dung system packages cua template:

```bash
cd /workspace/AIC26
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,api,semantic]"
python -m pip install boto3
```

Neu thieu `ffmpeg`, `git`, `tmux` hoac cong cu build:

```bash
apt-get update
apt-get install -y ffmpeg git tmux build-essential python3-venv
```

Kiem tra Python va CUDA:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Gia tri cuoi phai la `True` neu backend se dung GPU.

## 6. Cai frontend

Vite 8 cua frontend can Node.js `20.19+` hoac `22.12+`. Neu template chua co
Node dung phien ban, cai Node 22 bang `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install 22
nvm use 22
```

Cai dependency va build kiem tra:

```bash
cd /workspace/AIC26/frontend
npm ci
npm run build
```

Khong can commit thu muc `frontend/dist/` duoc tao ra boi lenh build.

## 7. Dua dataset va index len Vast

### Cach A: rsync tu may ca nhan

Chay cac lenh nay tren may ca nhan, khong phai ben trong Vast:

```bash
rsync -azh --info=progress2 \
  -e "ssh -p <SSH_PORT>" \
  /Users/<USER>/AIC26/data/raw/ \
  root@<VAST_IP>:/workspace/aic-data/
```

Neu da co FAISS index:

```bash
rsync -azh --info=progress2 \
  -e "ssh -p <SSH_PORT>" \
  /Users/<USER>/AIC26/artifacts/indexes/aic_l21_demo/ \
  root@<VAST_IP>:/workspace/aic-index/
```

### Cach B: tai truc tiep tu R2 tren Vast

Khong can tai data qua may ca nhan. Voi corpus lon, dung R2 lam source of truth
va tai truc tiep vao Vast Volume hoac container disk. Repo da co R2 CLI va tai
lieu chi tiet tai `docs/R2_DATA_MANAGEMENT.md`.

```bash
cd /workspace/AIC26
cp .env.example .env
```

Nhap R2 credentials vao `.env`, sau do nap cac bien vao shell:

```bash
source .venv/bin/activate
set -a
source .env
set +a
```

Cai `boto3` neu chua cai va dung `list` de kiem tra quyen truy cap cung layout
thuc te truoc khi tai:

```bash
python -m pip install boto3
python -m mvr.cli r2 list data/corpus/keyframes --max-keys 5
python -m mvr.cli r2 list data/corpus/indexes --max-keys 20
python -m mvr.cli r2 list data/raw --max-keys 20
```

Tao runtime root tren Vast. Nen dat thu muc nay tren Vast Volume neu muon giu
data sau khi destroy instance:

```bash
mkdir -p /workspace/r2-runtime
```

Tai toan bo canonical corpus, gom keyframe, metadata, embeddings va indexes:

```bash
python -m mvr.cli r2 pull \
  --corpus \
  --local-root /workspace/r2-runtime
```

Lenh tren khong tai raw videos, vi video nam trong `data/raw/`. De tai mot demo
subset:

```bash
python -m mvr.cli r2 pull \
  --videos L21_V001,L21_V002 \
  --batch 1 \
  --collection L21 \
  --local-root /workspace/r2-runtime
```

De tai ca collection video neu R2 dung canonical prefix:

```bash
python -m mvr.cli r2 pull \
  --prefix data/raw/1/videos/L21 \
  --local-root /workspace/r2-runtime
```

Sau khi tai, du lieu nam theo full R2 key:

```text
/workspace/r2-runtime/data/corpus/keyframes/
/workspace/r2-runtime/data/corpus/indexes/
/workspace/r2-runtime/data/raw/1/videos/L21/
```

Backend hien tai doc keyframe, video, mapping CSV va FAISS index tu filesystem.
Tao dataset root gon bang symlink:

```bash
mkdir -p /workspace/aic-data
ln -s /workspace/r2-runtime/data/corpus/keyframes \
  /workspace/aic-data/keyframes
ln -s /workspace/r2-runtime/data/raw/1/videos/L21 \
  /workspace/aic-data/videos
```

Tim mapping CSV va FAISS manifest trong corpus vua tai:

```bash
find /workspace/r2-runtime -type f -path '*map_keyframes*' | head
find /workspace/r2-runtime/data/corpus/indexes \
  -type f -name manifest.json -print
```

Tao symlink cho mapping, thay `<R2_MAPPING_DIRECTORY>` bang thu muc tim thay:

```bash
ln -s <R2_MAPPING_DIRECTORY> /workspace/aic-data/map_keyframes
```

Lien ket thu muc chua `manifest.json` cua FAISS index vao runtime path ma cac
buoc sau su dung. Thay `aic_l21` bang ten index thuc te tren R2:

```bash
ln -s /workspace/r2-runtime/data/corpus/indexes/aic_l21 \
  /workspace/aic-index
```

Neu R2 chua co `map_keyframes` hoac FAISS index, can upload cac artifact nay len
R2 mot lan hoac build index tren Vast theo buoc 9. Khong commit `.env` va khong
chia se R2 secret key.

## 8. Kiem tra layout dataset

Dataset root dung cho app phai co dang:

```text
/workspace/aic-data/
|- videos/
|  `- L21_V001.mp4
|- keyframes/
|  `- L21_V001/
|     |- 001.jpg
|     `- ...
`- map_keyframes/
   `- L21_V001.csv
```

Kiem tra:

```bash
find /workspace/aic-data/keyframes -type f | head
find /workspace/aic-data/videos -type f | head
find /workspace/aic-data/map_keyframes -type f | head
```

Neu bo du lieu dang dung ten `video` va `map-keyframes`, tao alias mot lan:

```bash
cd /workspace/aic-data
[ -e videos ] || ln -s video videos
[ -e map_keyframes ] || ln -s map-keyframes map_keyframes
```

## 9. Build FAISS index neu chua co

Bo qua buoc nay neu `/workspace/aic-index` da co index hoan chinh.

```bash
cd /workspace/AIC26
source .venv/bin/activate

python -m mvr.cli build-aic-index \
  --dataset-root /workspace/aic-data \
  --output /workspace/aic-index \
  --device cuda
```

Kiem tra artifact:

```bash
find /workspace/aic-index -maxdepth 1 -type f -print
```

Index hop le thuong co `keyframes.faiss`, `metadata.json`, `manifest.json` va
file embedding lien quan.

## 10. Chay backend

Tao tmux session de backend khong tat khi mat ket noi SSH:

```bash
tmux new -s aic-backend
```

Trong tmux session:

```bash
cd /workspace/AIC26
source .venv/bin/activate

export KMP_DUPLICATE_LIB_OK=TRUE
export MVR_SEMANTIC_BACKEND=faiss_reference
export MVR_SEMANTIC_INDEX=/workspace/aic-index
export MVR_AIC_DATASET_ROOT=/workspace/aic-data
export MVR_KEYFRAME_DIR=/workspace/aic-data/keyframes
export MVR_VIDEO_DIR=/workspace/aic-data/videos
export MVR_DEVICE=cuda
export PYTHONPATH=src

python -m uvicorn mvr.api.app:app \
  --host 127.0.0.1 \
  --port 8000
```

Nhan `Ctrl+B`, sau do nhan `D` de detach khoi tmux.

Kiem tra backend ngay tren Vast:

```bash
curl http://127.0.0.1:8000/health
```

Ket qua can co:

```json
{
  "status": "ok",
  "api": "ready",
  "semantic": {
    "configured": true,
    "available": true
  }
}
```

## 11. Chay frontend

Tao tmux session thu hai:

```bash
tmux new -s aic-frontend
```

Trong session:

```bash
cd /workspace/AIC26/frontend
export VITE_API_BASE_URL=http://127.0.0.1:8000
npm run dev -- --host 127.0.0.1 --port 5173
```

Detach bang `Ctrl+B`, sau do `D`.

Kiem tra hai session:

```bash
tmux ls
```

Xem lai log truc tiep:

```bash
tmux attach -t aic-backend
tmux attach -t aic-frontend
```

## 12. Mo app tu may ca nhan

Chay lenh sau tren may ca nhan va giu terminal nay mo:

```bash
ssh -N -p <SSH_PORT> root@<VAST_IP> \
  -L 5173:127.0.0.1:5173 \
  -L 8000:127.0.0.1:8000
```

Mo trinh duyet:

```text
http://127.0.0.1:5173
```

SSH tunnel cho phep dung app ma khong public truc tiep port backend ra Internet.
Moi thanh vien can tao tunnel tu may cua ho neu muon truy cap cung Vast instance.

## 13. Test search that

Kiem tra API truc tiep tu may ca nhan sau khi tunnel da mo:

```bash
curl http://127.0.0.1:8000/health
```

Sau do vao UI, nhap mot query tieng Anh phu hop voi subset da index. Neu search
tra ve ket qua nhung anh/video khong hien thi, kiem tra tiep:

```bash
curl -I http://127.0.0.1:8000/keyframes/<VIDEO_ID>/<KEYFRAME_FILE>.jpg
curl -I http://127.0.0.1:8000/videos/<VIDEO_ID>.mp4
```

## 14. Cap nhat code tren Vast

```bash
cd /workspace/AIC26
git switch feat/backend-ui-api
git pull --ff-only origin feat/backend-ui-api

source .venv/bin/activate
python -m pip install -e ".[dev,api,semantic]"

cd frontend
npm ci
npm run build
```

Khoi dong lai cac tmux session sau khi cap nhat code.

## 15. Xu ly loi thuong gap

### Port da duoc su dung

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:5173 -sTCP:LISTEN
```

Neu `/health` van tra ve `status: ok`, backend cu dang chay va khong can mo them
mot Uvicorn process tren cung port.

### Health bao semantic unavailable

Kiem tra index path va file:

```bash
echo "$MVR_SEMANTIC_INDEX"
find /workspace/aic-index -maxdepth 1 -type f -print
```

Dam bao backend duoc khoi dong trong cung shell co cac bien moi truong o buoc
10.

### Search khong co ket qua

Kiem tra:

1. Query phu hop voi video da duoc index.
2. `MVR_SEMANTIC_INDEX` tro dung index.
3. Dataset va index duoc tao tu cung keyframe set.
4. `MVR_AIC_DATASET_ROOT` co `map_keyframes/` de map ve original frame.

### Co ket qua nhung khong co anh

```bash
find /workspace/aic-data/keyframes -type f | head
```

Dam bao `MVR_KEYFRAME_DIR=/workspace/aic-data/keyframes` khi khoi dong backend.

### Xem duoc keyframe nhung khong xem duoc video

```bash
find /workspace/aic-data/videos -type f | head
```

Ten video phai khop `video_id`, vi du `L21_V001.mp4`, va backend phai co:

```bash
export MVR_VIDEO_DIR=/workspace/aic-data/videos
```

### CUDA khong kha dung

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Co the chay tam bang CPU:

```bash
export MVR_DEVICE=cpu
```

Search van hoat dong, nhung load model va truy van se cham hon.

## 16. Checklist hoan tat

- [ ] `nvidia-smi` nhan GPU.
- [ ] Backend dependencies da cai.
- [ ] `npm ci` va `npm run build` thanh cong.
- [ ] Dataset co `videos/`, `keyframes/`, `map_keyframes/`.
- [ ] FAISS index ton tai va khop dataset.
- [ ] `/health` bao semantic configured/available.
- [ ] Backend tmux dang chay port `8000`.
- [ ] Frontend tmux dang chay port `5173`.
- [ ] SSH tunnel dang mo.
- [ ] UI search ra keyframe that.
- [ ] Mo duoc video va Video Summary.
