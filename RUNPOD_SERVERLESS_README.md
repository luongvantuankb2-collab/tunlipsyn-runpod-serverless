# Tun Lipsync RunPod Serverless Ready

Muc tieu: web UI/backend chay thuong truc, GPU RunPod Serverless chi chay khi co job render.

## Thu muc

- `web_saas/backend`: web UI + API + queue + dispatch RunPod.
- `web_saas/worker/serverless_handler.py`: RunPod Serverless handler, xu ly dung 1 job.
- `web_saas/worker/Dockerfile.runpod`: Docker image cho GPU worker.
- `web_saas/.env.runpod.example`: env mau cho web backend.
- `setup_web_serverless.sh`: cai web backend tren VPS/Linux.
- `start_web_serverless.sh`: chi start web, khong start GPU worker local.
- `build_runpod_image.ps1`: build/push Docker image tu Windows.

## Buoc 1: build Docker image worker

Chay tren Windows PowerShell, may can co Docker Desktop va da login registry:

```powershell
cd D:\tunlipsyn_runpod_serverless_ready
.\build_runpod_image.ps1 -Image YOUR_DOCKER_USER/tunlipsyn-runpod-worker:v1
```

Neu dung GHCR:

```powershell
.\build_runpod_image.ps1 -Image ghcr.io/YOUR_USER/tunlipsyn-runpod-worker:v1
```

## Buoc 2: tao RunPod Serverless endpoint

Trong RunPod:

- Type: Serverless endpoint
- Worker image: image vua push
- GPU: RTX 4090 truoc
- Workers min: `0`
- Workers max: `1` luc test
- Idle timeout: de mac dinh hoac thap
- Request timeout: 900-1800 giay
- Environment:

```text
WORK_DIR=/workspace/work
RENDER_BACKEND=runpod-serverless
LATENTSYNC_FORCE_STABLE_PROFILE=1
LATENTSYNC_MODEL_PRESET=v15
LATENTSYNC_STEPS=40
LATENTSYNC_GUIDANCE=1.8
LATENTSYNC_CROP_SCALE=0.75
```

Sau khi tao endpoint, lay `Endpoint ID`.

## Buoc 3: cai web backend

Tren VPS/Linux hoac pod CPU:

```bash
cd /workspace/tunlipsyn_runpod_serverless_ready
bash setup_web_serverless.sh
nano web_saas/.env
```

Sua cac dong:

```text
PUBLIC_BASE_URL=http://YOUR_WEB_HOST:8080
WORKER_BASE_URL=http://YOUR_WEB_HOST:8080
RUNPOD_CALLBACK_BASE_URL=http://YOUR_WEB_HOST:8080
WORKER_TOKEN=mot-chuoi-bi-mat-dai
RUNPOD_DISPATCH_ENABLED=1
RUNPOD_API_KEY=runpod_api_key_cua_ban
RUNPOD_ENDPOINT_ID=endpoint_id_cua_ban
```

Chay web:

```bash
mkdir -p logs
nohup bash start_web_serverless.sh > logs/web.log 2>&1 &
sleep 5
curl -s http://127.0.0.1:8080/api/health
```

Mo:

```text
http://YOUR_WEB_HOST:8080
```

## Buoc 4: test job

Tao job tren web. Log dung se co cac dong:

```text
RunPod serverless job dispatched
RunPod worker claimed job
Preflight: LatentSync v1.5 256 / 8GB
LatentSync params: steps=40, guidance=1.8, crop=0.75
Worker uploaded result
```

Neu job dung o `queued`, kiem tra:

```bash
tail -f logs/web.log
```

Neu RunPod khong goi duoc backend, kiem tra `RUNPOD_CALLBACK_BASE_URL` phai la URL public ma worker RunPod truy cap duoc.

## Ghi chu quan trong

- Khong dung `start_tunlipsyn_ezycloud.sh` cho Serverless vi file do start worker polling local.
- Dung `start_web_serverless.sh`.
- Luc khong co job, GPU RunPod Serverless co the scale ve 0.
- File input/output hien dang luu tren web backend local storage. Sau khi MVP on dinh, co the chuyen sang Cloudflare R2.
- File de mo dau tien sang mai: `TOMORROW_START_HERE.txt`.
