# AGENT GUIDE — ResNet50 phân loại giới tính (FairFace full) trên Kaggle **2× T4 (DDP)**

> Bản này **thay thế** bản single-GPU. Spec cho coding agent: đọc hết trước khi code. Output là **1 Kaggle notebook** + 2 file Python do notebook ghi ra (`ff_common.py`, `train_ddp.py`), train **4 model ResNet50** tuần tự bằng `torchrun` trên **2 GPU T4**, so sánh công bằng.
> Không hỏi lại user những gì đã ghi ở đây. Chỗ nào ghi "agent tự quyết" thì tự chọn và ghi lại vào log/markdown cuối notebook.

---

## 0. Tổng quan & ràng buộc cứng

| Hạng mục | Quy định |
|---|---|
| Task | Phân loại nhị phân **Male / Female** từ ảnh mặt |
| Dataset | **FairFace bản full = margin 1.25** (ảnh có thêm bối cảnh quanh mặt: tóc, cổ, vai). Có flag `FAIRFACE_MARGIN` ("1.25" mặc định, "0.25" để đổi) |
| Model | 4 model ResNet50, tăng dần độ nâng cấp (§6) |
| GPU | **2× T4 (Kaggle "GPU T4 x2")**, **DDP + NCCL**, fp16 AMP (T4 không có bf16/TF32) |
| Dữ liệu | **KHÔNG gắn `/kaggle/input`**. Tải về `/kaggle/working` (local disk), build cache, rồi mới train |
| Loader | Không dùng `DataLoader` + decode JPEG (chỉ ~4 CPU core). Dùng **uint8 memmap + GPU augmentation**, mỗi rank 1 thread prefetch |
| Reproducibility | Seed 42, cùng split, cùng ngân sách epoch cho mọi model |
| Giới hạn Kaggle | Session ≤ 12h, disk `/kaggle/working` ≤ 20GB, RAM ~29GB, **resume** bắt buộc |

**Về RetinaFace:** FairFace đã crop mặt sẵn → mặc định KHÔNG chạy RetinaFace. Chỉ để cờ `USE_RETINAFACE=False`, không implement trừ khi user bảo.

**Nguyên tắc kiến trúc code (rất quan trọng với DDP):**
- **Notebook process KHÔNG được khởi tạo CUDA** (không gọi `torch.cuda.get_device_name`, `.cuda()`, ...; dùng `!nvidia-smi` để xem GPU). Mọi việc dùng GPU (sanity, benchmark, train, eval) chạy trong tiến trình `torchrun`. Nếu không, notebook giữ context + bộ nhớ trên GPU0 và làm lệch bộ nhớ giữa 2 GPU.
- Phần xử lý dữ liệu (tải, build cache, split) và phần báo cáo (đọc json/csv, vẽ hình) chạy ở notebook, **chỉ dùng CPU**.
- Code GPU nằm trong 2 file ghi bằng `%%writefile`: `ff_common.py` (data/model/aug/EMA/eval) và `train_ddp.py` (entry point, có các mode `--model mX`, `--sanity`, `--bench`).

---

## 1. Cài đặt notebook Kaggle

- Accelerator: **GPU T4 x2**. **Internet: ON** (account đã verify phone). Không attach dataset vào Input.
- Cell đầu (chỉ CPU):

```python
import os, time, json, math, shutil, subprocess, sys
import numpy as np

SEED = 42
FAIRFACE_MARGIN = "1.25"            # "1.25" = bản full (mặc định) | "0.25"
STORE_SIZE = 224
ROOT   = "/kaggle/working"
CACHE  = f"{ROOT}/fairface_{FAIRFACE_MARGIN}_cache"
CKPT   = f"{ROOT}/ckpt"
RESULT = f"{ROOT}/results"
for d in (CACHE, CKPT, RESULT): os.makedirs(d, exist_ok=True)
CLEANUP_CACHE = True
USE_RETINAFACE = False
MODELS_TO_RUN = ["m1", "m2", "m3", "m4"]
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)   # kỳ vọng thấy 2 GPU
```

**Cảnh báo commit:** *Save Version → Save & Run All* coi mọi thứ trong `/kaggle/working` là output; cache ~15GB sẽ làm commit chậm/fail → cell cuối xóa `CACHE` khi `CLEANUP_CACHE=True` (giữ `ckpt/`, `results/`, 2 file `.py`). (Hoặc đặt `CACHE` ở `/kaggle/temp/...`; agent tự chọn, mặc định `/kaggle/working` theo yêu cầu user.)

---

## 2. Lấy dữ liệu về `/kaggle/working`

Thử lần lượt A → B → C. **Không bịa slug/URL**; mỗi bước `print` kết quả kiểm tra (số mẫu, features, kích thước ảnh) rồi mới đi tiếp.

**A. Hugging Face (ưu tiên)** — config theo margin:
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceM4/FairFace", FAIRFACE_MARGIN, cache_dir=f"{ROOT}/hf_cache")
print(ds); print(ds["train"].features)     # kỳ vọng train ≈ 86,744 ; validation ≈ 10,954
print(ds["train"][0]["image"].size)        # IN RA kích thước thật của ảnh margin này
```
Lỗi do dataset script / version `datasets` → thử `snapshot_download(repo_id="HuggingFaceM4/FairFace", repo_type="dataset", revision="refs/convert/parquet", ...)` rồi `load_dataset("parquet", data_files=...)`.

**B. Kaggle (kagglehub):** `os.environ["KAGGLEHUB_CACHE"]=f"{ROOT}/kagglehub"`, `kagglehub.dataset_download("<owner>/<slug>")`; slug phải được xác nhận, không đoán. Phải chọn đúng bản margin 1.25.

**C. Repo gốc github.com/joojs/fairface:** zip ảnh margin 1.25 + `fairface_label_train.csv`, `fairface_label_val.csv` (cột `file, age, gender, race, service_test`) trên Google Drive → `gdown`; kiểm tra link trong README.

**Quy ước split (mọi model dùng chung):**
- `train_all` = split train chính thức (86,744 ảnh).
- **Dev 10%** tách từ `train_all`, **stratified theo (race, gender)**, seed 42 → chọn checkpoint tốt nhất. Lưu `dev_idx.npy`, `trainmain_idx.npy`.
- **Test = split validation chính thức** (10,954 ảnh), chỉ eval **1 lần/model** bằng checkpoint chọn trên dev. Không chọn model/hyperparameter bằng test.

**Label:** `Female = 1`, `Male = 0`; lấy index từ `features["gender"].names` (assert có "Male","Female"), không hardcode. Lưu thêm `race` (int8) để eval theo nhóm; lưu `race_names` vào `CACHE/meta.json`.

---

## 3. Build cache uint8 (1 lần, ở notebook, CPU)

- `train_x.npy`: `(86744, 224, 224, 3) uint8` ≈ **13.1GB**; `test_x.npy` ≈ 1.65GB (`np.lib.format.open_memmap(mode="w+")`).
- Margin 1.25 có thể có kích thước gốc khác 224 → luôn **resize về `STORE_SIZE=224`** (bilinear/area) lúc build cache. **Không** tăng `STORE_SIZE` lên 256+ (cache > 19GB, vượt disk).
- Decode bằng `ThreadPoolExecutor(4)` theo chunk ~2048.
- Lưu `train_gender.npy` (float32 0/1), `train_race.npy` (int8), tương tự `test_*`.
- Xong → **xóa raw/HF cache** (`shutil.rmtree(f"{ROOT}/hf_cache")`). In `du` của `CACHE` + `shutil.disk_usage`.
- Nếu tổng disk > ~17GB hoặc RAM áp lực → fallback `STORE_SIZE=176`, ghi rõ vào log.

```python
def build_cache(split, name, size=224, chunk=2048):
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    N = len(split)
    X = np.lib.format.open_memmap(f"{CACHE}/{name}_x.npy", mode="w+", dtype=np.uint8, shape=(N, size, size, 3))
    def dec(im):
        im = im.convert("RGB")
        if im.size != (size, size): im = im.resize((size, size), Image.BILINEAR)
        return np.asarray(im)
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, N, chunk):
            imgs = split[i:i+chunk]["image"]
            X[i:i+len(imgs)] = np.stack(list(ex.map(dec, imgs)))
    X.flush()
```

Sau đó mở bằng `np.load(path, mmap_mode="r")` trong từng rank. **Hai rank dùng chung page cache của OS** nên 13GB chỉ chiếm RAM một lần.

---

## 4. Ghi `ff_common.py` — data pipeline (rank-aware)

### 4.1 Loader: thread prefetch, shard theo rank

```python
import os, math, threading, queue, copy, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class MemmapLoader:
    """Mỗi rank chạy 1 instance. Thứ tự shuffle sinh từ seed+epoch (GIỐNG nhau ở mọi rank),
    sau đó mỗi rank lấy order[rank::world]. Train: drop_last=True để số step/rank bằng nhau (tránh deadlock)."""
    def __init__(self, X, y, bs, indices=None, shuffle=False, drop_last=False,
                 prefetch=4, seed=0, rank=0, world=1):
        self.X, self.y, self.bs = X, y, bs
        self.indices = np.arange(len(y)) if indices is None else np.asarray(indices)
        self.shuffle, self.drop_last, self.prefetch = shuffle, drop_last, prefetch
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, 0
    def set_epoch(self, e): self.epoch = e
    def _order(self):
        order = self.indices
        if self.shuffle:
            order = np.random.default_rng(self.seed + self.epoch).permutation(order)
        if self.world > 1:
            n = (len(order) // self.world) * self.world
            order = order[:n][self.rank::self.world]
        return order
    def __len__(self):
        n = len(self.indices) // self.world if self.world > 1 else len(self.indices)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)
    def __iter__(self):
        order = self._order(); q = queue.Queue(self.prefetch)
        def worker():
            for i in range(0, len(order), self.bs):
                ids = order[i:i+self.bs]
                if self.drop_last and len(ids) < self.bs: break
                ids = np.sort(ids)                                        # đọc memmap theo thứ tự tăng
                q.put((torch.from_numpy(self.X[ids]).pin_memory(), torch.from_numpy(self.y[ids])))
            q.put(None)
        threading.Thread(target=worker, daemon=True).start()
        while (item := q.get()) is not None:
            xb, yb = item
            yield xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
```
Dev/test eval chỉ chạy ở rank 0 → tạo loader với `world=1`.

### 4.2 GPU augmentation (1 `grid_sample`/batch, đồng thời resize → dùng cho progressive resizing)

```python
def make_norm(device):
    m = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    s = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    return m, s

def gpu_prep(xb, out_size, train, aug, norm):
    MEAN, STD = norm
    x = xb.permute(0,3,1,2).float().div_(255)         # NHWC uint8 -> NCHW (stride đã là channels_last)
    B = x.size(0)
    if train and aug is not None:
        dev = x.device
        s    = torch.empty(B, device=dev).uniform_(*aug["scale"]).sqrt()
        ang  = torch.empty(B, device=dev).uniform_(-aug["rot"], aug["rot"]) * math.pi/180
        sign = (torch.rand(B, device=dev) < 0.5).float()*2 - 1                  # hflip
        tx = (torch.rand(B, device=dev)*2-1)*(1-s); ty = (torch.rand(B, device=dev)*2-1)*(1-s)
        c_, s_ = torch.cos(ang), torch.sin(ang)
        theta = torch.stack([torch.stack([s*c_*sign, -s*s_, tx], 1),
                             torch.stack([s*s_*sign,  s*c_, ty], 1)], 1)
        grid = F.affine_grid(theta, (B,3,out_size,out_size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        c = aug["color"]
        if c > 0:
            b  = 1 + (torch.rand(B,1,1,1, device=dev)*2-1)*c
            ct = 1 + (torch.rand(B,1,1,1, device=dev)*2-1)*c
            m  = x.mean((1,2,3), keepdim=True)
            x  = (((x-m)*ct + m)*b).clamp_(0,1)
    elif x.size(-1) != out_size:
        x = F.interpolate(x, size=(out_size,out_size), mode="bilinear", antialias=True, align_corners=False)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

AUG_BASIC = dict(scale=(0.8,1.0), rot=0,  color=0.0)    # + hflip luôn có
AUG_STD   = dict(scale=(0.7,1.0), rot=10, color=0.2)
```
Với margin 1.25 mặt nằm nhỏ giữa khung; **đừng crop quá mạnh**: giữ `scale` min ≥ 0.7 như trên.

**Sanity (mode `--sanity`, rank 0):** lưu lưới 16 ảnh sau aug + nhãn ra `results/aug_check.png` để nhìn bằng mắt.

---

## 5. Checklist tối ưu tốc độ (2× T4)

**Mỗi GPU:**
1. AMP fp16 + `GradScaler`. Không dùng bf16.
2. `channels_last` cho model và input.
3. `cudnn.benchmark=True`.
4. `torch.optim.AdamW(..., fused=True)`; `zero_grad(set_to_none=True)`.
5. Không `.item()`/`print` mỗi step; cộng dồn loss trên GPU, `all_reduce` cuối epoch.
6. Pinned memory + `non_blocking=True`; prefetch depth 4; aug trên GPU.
7. `torch.compile`: **mặc định OFF** với DDP (rủi ro/recompile). Chỉ benchmark thử nếu còn thời gian; không dùng với progressive resizing (M4).

**DDP:**
8. **Batch size theo GPU mặc định 128** → global 256 (các lr trong §6 đã tune cho global bs 256). Benchmark `{96,128,192}`/GPU; chỉ đổi nếu nhanh hơn ≥8%; nếu global bs đổi thì scale lr theo `sqrt(global/256)`.
9. `DDP(model, device_ids=[local], gradient_as_bucket_view=True)`; giữ `broadcast_buffers=True` (mặc định) để BN running stats của mọi rank trùng rank 0.
10. **Không dùng SyncBN** (chậm; bs 128/GPU đủ cho BN).
11. **Không dùng `nn.DataParallel`.**
12. Mỗi rank 1 loader thread; `OMP_NUM_THREADS=2` khi launch (tránh tranh CPU giữa 2 rank).
13. Eval dev/test chạy **chỉ rank 0** trên **module gốc (không qua DDP wrapper)** → các rank khác đợi ở `dist.barrier()`. Eval ~10–20s/epoch, chấp nhận.
14. T4 x2 của Kaggle nối PCIe (không NVLink) → all-reduce 100MB grads/step có overhead nhỏ. Kỳ vọng scaling **~1.7–1.9×**.
15. **Mode `--bench`** (synthetic, 60 step sau 15 step warmup): chạy `torchrun --nproc_per_node=1` rồi `--nproc_per_node=2`, in img/s và **scaling efficiency** = `ips2 / (2*ips1)`. Nếu < 0.8 → xem §10 (NCCL env). Lưu `results/benchmark.json`. Cell này < 4 phút.

Ước lượng thô để làm mốc (phải đo thật): ~350–450 img/s/GPU ở 224px → ~700–800 img/s tổng → **~1.8–2.2 phút/epoch** trên 78k ảnh train_main. 128px nhanh hơn ~2.5–3×.

---

## 6. Bốn model (ladder cộng dồn)

Chung: `BCEWithLogits` trên **1 logit** (output shape `(B,)`), threshold 0.5, cosine per-step, warmup 1 epoch ở stage 2, chọn best theo **dev accuracy** (tie → dev loss thấp hơn). Hyperparameter dưới đây ứng với **global bs 256**.

| | **M1 Baseline** | **M2 + Recipe** | **M3 + Architecture** | **M4 + Efficiency & Data tricks** |
|---|---|---|---|---|
| Pretrained | torchvision `IMAGENET1K_V1` | V1 | V1 | **`IMAGENET1K_V2`** |
| Kiến trúc | ResNet50 + `fc→Linear(2048,1)` | = M1 | + CBAM gate mỗi stage + GeM + MLP head | = M3 |
| Huấn luyện | full fine-tune 1 giai đoạn | **2 giai đoạn** + **discriminative LR** | = M2 | = M2 |
| Epoch | 12 | 2 + 10 | 2 + 10 | 2 + 12 (progressive) |
| Optimizer | AdamW lr 2e-4, wd 0.01 | AdamW base 4e-4 (mults), wd 0.02 | = M2 | = M2 |
| Aug | `AUG_BASIC` | `AUG_STD` | `AUG_STD` | `AUG_STD` + mixup |
| Label smoothing | 0 | 0.05 | 0.05 | 0.05 |
| EMA | – | – | có (0.998) | có |
| Resolution | 224 | 224 | 224 | **progressive 128→176→224** |
| Mixup | – | – | – | p=0.5, α=0.2, **tắt 3 epoch cuối** |
| TTA lúc test | – | – | – | **hflip** (trung bình logit) |

**Model chung** (M1 = mọi flag tắt, tương đương `torchvision.resnet50` + fc mới):
```python
import torchvision
class CBAMGate(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c//r, 1), nn.ReLU(inplace=True), nn.Conv2d(c//r, c, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)
        self.alpha = nn.Parameter(torch.zeros(1))                 # =0 -> identity lúc khởi tạo
    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2,3), keepdim=True)) + self.mlp(x.amax((2,3), keepdim=True)))
        y = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([y.mean(1, keepdim=True), y.amax(1, keepdim=True)], 1)))
        y = y * sa
        return x + self.alpha * (y - x)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__(); self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return x.float().clamp(min=self.eps).pow(self.p).mean((2,3)).pow(1.0/self.p)

class ResNet50Gender(nn.Module):
    def __init__(self, weights="V1", attn=False, gem=False, mlp_head=False, drop=0.3):
        super().__init__()
        W = torchvision.models.ResNet50_Weights
        b = torchvision.models.resnet50(weights=W.IMAGENET1K_V1 if weights=="V1" else W.IMAGENET1K_V2)
        self.stem   = nn.Sequential(b.conv1, b.bn1, b.relu, b.maxpool)
        self.layers = nn.ModuleList([b.layer1, b.layer2, b.layer3, b.layer4])
        self.gates  = nn.ModuleList([CBAMGate(c) for c in (256,512,1024,2048)]) if attn else None
        self.pool   = GeM() if gem else nn.AdaptiveAvgPool2d(1)
        self.head   = (nn.Sequential(nn.Dropout(drop), nn.Linear(2048,512), nn.GELU(), nn.Dropout(drop), nn.Linear(512,1))
                       if mlp_head else nn.Linear(2048,1))
    def forward(self, x):
        x = self.stem(x)
        for i, l in enumerate(self.layers):
            x = l(x)
            if self.gates is not None: x = self.gates[i](x)
        return self.head(self.pool(x).flatten(1)).squeeze(1)
```
Tải pretrained weights: để **rank 0 tải trước** (hoặc `dist.barrier()` sau khi rank 0 build) để tránh 2 rank cùng ghi cache torch hub. Hoặc tải 1 lần ở notebook (CPU, không đụng CUDA).

**Discriminative LR (M2/M3/M4)** — `base_lr = 4e-4`:

| Nhóm | mult | lr |
|---|---|---|
| `stem` + `layers.0` | 0.1 | 4e-5 |
| `layers.1` | 0.25 | 1e-4 |
| `layers.2` | 0.5 | 2e-4 |
| `layers.3` | 1.0 | 4e-4 |
| `gates`, `pool`, `head` | 2.5 | 1e-3 |

Weight decay chỉ cho tham số `ndim > 1` (không decay bias, BN, `alpha`, GeM `p`).

**Stage 1 (M2/M3/M4, 2 epoch):** freeze `stem` + `layers` (`requires_grad=False`) và **giữ BN phần đóng băng ở eval mode** (sau `model.train()` gọi `module.stem.eval(); module.layers.eval()`). Chỉ train `head` (+`gates`,`pool` ở M3/M4), lr 1e-3, không warmup. M4: stage 1 chạy @128. Sang stage 2: unfreeze, **tạo lại optimizer + scheduler + DDP wrapper** (§7).

**EMA (M3/M4)** — bọc **module gốc**, không phải DDP wrapper:
```python
class EMA:
    def __init__(self, model, decay=0.998):
        self.m = copy.deepcopy(model).eval(); self.d = decay
        for p in self.m.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        ep, mp = list(self.m.parameters()), [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ep, self.d); torch._foreach_add_(ep, mp, alpha=1-self.d)
        for be, bm in zip(self.m.buffers(), model.buffers()): be.copy_(bm)
```
Mỗi rank giữ 1 EMA riêng; vì params đã đồng bộ nên EMA các rank giống nhau. Bắt đầu update từ đầu stage 2. Eval/best/test dùng EMA.

**Progressive resizing (M4):** stage 2 (12 epoch) = 4 epoch @128, 4 @176, 4 @224 (đổi `out_size` trong `gpu_prep`; cache vẫn 224). Dev/test luôn eval ở 224.

**Mixup (M4):** mỗi batch với prob 0.5: `lam~Beta(0.2,0.2)` (sinh ở rank, khác nhau giữa rank cũng được), `x = lam*x + (1-lam)*x[perm]`, target mềm tương ứng. Không tính train-acc ở batch mixup. Tắt 3 epoch cuối.

**TTA (M4, test):** `logit = (f(x) + f(flip(x)))/2`.

---

## 7. Ghi `train_ddp.py` — quy tắc DDP

### 7.1 Khởi tạo
```python
import datetime, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def ddp_init(seed=42):
    rank  = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    torch.backends.cudnn.benchmark = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)  # CÙNG seed mọi rank khi build model
    return rank, world, local
# sau khi build model + DDP: torch.manual_seed(seed + 1000*rank)  -> aug/mixup khác nhau giữa các rank
```
DDP broadcast params từ rank 0 lúc khởi tạo nên head init giống nhau.

### 7.2 Wrap theo stage (freeze/unfreeze)
DDP chỉ theo dõi param `requires_grad=True` tại thời điểm tạo. Vì vậy mỗi stage: **set `requires_grad` trước → tạo DDP**. Khi sang stage 2: `del ddp_model; gc.collect(); torch.cuda.empty_cache()` → unfreeze → tạo DDP mới → tạo optimizer/scheduler mới.
- **Test bắt buộc trong `--sanity`:** chạy 3 step stage 1 → re-wrap → 3 step stage 2, không treo/lỗi.
- **Fallback nếu re-wrap lỗi/treo:** tạo **1 DDP duy nhất ngay từ đầu** với `find_unused_parameters=True` (mọi param `requires_grad=True` lúc tạo), rồi freeze/unfreeze bằng `requires_grad_` ở từng stage. Chậm hơn chút ở stage 1 nhưng chạy chắc.

### 7.3 Skeleton vòng lặp
```python
for ep in range(start_ep, total_ep):
    train_loader.set_epoch(ep)                       # cùng seed+epoch ở mọi rank
    ddp_model.train(); apply_bn_freeze_if_stage1(module)
    loss_sum = torch.zeros((), device="cuda"); cnt = torch.zeros((), device="cuda")
    for xb, yb in train_loader:                      # số step/rank bằng nhau (drop_last=True)
        x = gpu_prep(xb, res, True, aug, norm); y = yb
        # (mixup nếu M4 và chưa vào 3 epoch cuối)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = ddp_model(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y*(1-ls) + 0.5*ls)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
        if ema: ema.update(module)
        loss_sum += loss.detach()*y.numel(); cnt += y.numel()
    if world > 1: dist.all_reduce(loss_sum); dist.all_reduce(cnt)
    if rank == 0:
        dev_metrics = evaluate(ema.m if ema else module, dev_loader)   # module GỐC, KHÔNG gọi ddp_model (tránh collective)
        log_row(...); save_best_if_improved(...); save_last(...)
    if world > 1: dist.barrier()
```
- Scheduler: `LambdaLR` per-step, `factor = min(1,(step+1)/warm) * 0.5*(1+cos(pi*progress))` (stage 1 không warmup). Tổng step tính theo `len(train_loader)` của **mỗi rank**.
- Chỉ **rank 0** in log, ghi CSV/JSON, lưu checkpoint.
- **Resume:** `ckpt/{m}_last.pt` (module, optimizer, scheduler, scaler, EMA, `stage`, `epoch`) lưu mỗi epoch bởi rank 0; khi start, mọi rank `torch.load(..., map_location=f"cuda:{local}")` sau `barrier`. Thứ tự data suy ra từ `seed+epoch` nên resume ở ranh giới epoch là nhất quán. Nếu `best.pt` + `{m}_metrics.json` đã có → skip model.
- GradScaler: mỗi rank 1 scaler; grads sau all-reduce giống nhau nên quyết định skip-step (inf) đồng nhất.
- Đo thời gian bằng `torch.cuda.synchronize()` ở đầu/cuối epoch; ghi `max_memory_allocated` (mỗi rank, lấy max).
- Cuối train (rank 0): load best → **eval test 1 lần** (M4 dùng TTA), lưu `results/{m}_metrics.json` và `results/{m}_test_logits.npy` (để notebook tính lại mọi thứ). Rank khác đợi ở `barrier`.

### 7.4 Launcher ở notebook
```python
def run(args):
    env = dict(os.environ, NCCL_IB_DISABLE="1", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    cmd = ["torchrun", "--standalone", "--nproc_per_node=2", "train_ddp.py"] + args
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout: print(line, end="")
    assert p.wait() == 0, "torchrun failed"

run(["--sanity"]); run(["--bench"])
for m in MODELS_TO_RUN: run(["--model", m])        # mỗi model 1 lần torchrun -> giải phóng sạch GPU giữa các model
```
Truyền `CACHE`, `CKPT`, `RESULT`, `FAIRFACE_MARGIN`, `BS_PER_GPU` qua argparse hoặc file `results/config.json`.

---

## 8. Đánh giá (mỗi model, trên test = FairFace val chính thức)

- Accuracy, F1 (positive = Female), ROC-AUC, confusion matrix, precision/recall từng lớp.
- **Accuracy theo 7 nhóm race** + `max_gap = max(acc) - min(acc)`; acc riêng Male/Female.
- **Hiệu năng:** params (M), tổng thời gian train (phút), img/s train trung bình (tổng 2 GPU), peak GPU mem (GB), latency inference (ms/ảnh bs=1; img/s bs=64; fp16, warmup, có sync) — đo ở rank 0 trên 1 GPU.
- Từ benchmark: img/s 1 GPU vs 2 GPU + scaling efficiency (đưa vào báo cáo).
- Notebook (CPU) gộp thành `results/summary.csv` + `results/summary.md`.

Hình trong `results/`: (1) loss & dev-acc theo epoch của 4 model, (2) bar test acc + train time, (3) bar acc theo race cho 4 model, (4) confusion matrix model tốt nhất, (5) `aug_check.png`, (6) bar so sánh img/s 1 GPU vs 2 GPU.

---

## 9. Cấu trúc notebook

1. Setup + config (CPU, §1)
2. Tải dataset margin 1.25 về `/kaggle/working` + verify (§2)
3. Build cache uint8 + split dev/test + xóa raw (§3)
4. `%%writefile ff_common.py` (§4, §6 model/EMA, evaluate)
5. `%%writefile train_ddp.py` (§7)
6. `run(["--sanity"])` (aug_check, overfit 1 batch, test re-wrap stage) → `run(["--bench"])` (1 vs 2 GPU) → chốt `BS_PER_GPU`
7. Train M1 → M2 → M3 → M4 (`run(["--model", m])`)
8. Tổng hợp bảng/hình (CPU)
9. Cleanup cache nếu `CLEANUP_CACHE` + liệt kê file output

---

## 10. Troubleshooting NCCL / Kaggle

- Treo ở `init_process_group` hoặc scaling < 0.8: thử lần lượt `NCCL_P2P_DISABLE=1`, rồi `NCCL_SHM_DISABLE=1`; đặt `NCCL_DEBUG=WARN` để xem log. Đo lại bằng `--bench` sau mỗi thay đổi, chọn env tốt nhất, ghi vào log.
- `torchrun --standalone` tự chọn port; nếu lỗi port thì thêm `--master_port=29511`.
- OOM ở 1 GPU: giảm `BS_PER_GPU` (và scale lr theo `sqrt`), không giảm độ phân giải.
- Nếu GPU0 dùng nhiều mem hơn GPU1 rõ rệt → kiểm tra notebook process có lỡ khởi tạo CUDA không (§0).
- Fallback cuối cùng nếu DDP không chạy được trên session: chạy **single-GPU** bằng `--nproc_per_node=1` (code phải chạy được với `WORLD_SIZE=1`, đã thiết kế sẵn) và ghi rõ vào báo cáo.

---

## 11. Lỗi hay gặp — tránh

- Số step khác nhau giữa các rank (quên `drop_last=True` ở train) → **deadlock**.
- Gọi `ddp_model(x)` trong eval ở rank 0 (collective không có đối tác) → treo. Eval chỉ gọi module gốc.
- Quên `dist.barrier()` sau phần việc rank 0 → rank khác chạy trước, lệch, timeout.
- Quên đổi `requires_grad` **trước** khi tạo DDP (hoặc không re-wrap sau khi unfreeze).
- Notebook process khởi tạo CUDA.
- Decode JPEG trong lúc train; đọc dữ liệu từ `/kaggle/input`.
- `.item()`/print mỗi step; dùng bf16 trên T4; dùng `DataParallel`; dùng SyncBN không cần thiết.
- Quên giữ BN ở eval khi stage 1.
- Chọn checkpoint/hyperparameter bằng test set; dev split không stratify.
- Để cache 15GB trong `/kaggle/working` rồi commit.
- So sánh không công bằng giữa các model (khác seed/split/epoch/bs).

---

## 12. Definition of Done

- [ ] Notebook chạy từ đầu đến cuối trên **T4 x2** bằng `torchrun`, dữ liệu margin 1.25 nằm hoàn toàn trên `/kaggle/working` (không attach input).
- [ ] `--bench` cho thấy scaling efficiency 2 GPU (mục tiêu ≥ 0.85; < 0.8 thì đã xử lý theo §10 và ghi lại).
- [ ] 4 model đều có `*_best.pt`, `*_log.csv`, `*_metrics.json`, `*_test_logits.npy`.
- [ ] `summary.csv/md` có: test acc, F1, AUC, per-race acc + gap, params, train time, img/s (2 GPU), peak mem, latency.
- [ ] `benchmark.json` + log nêu `BS_PER_GPU` và NCCL env đã chọn + lý do.
- [ ] Đủ 6 hình ở §8; `aug_check.png` đã kiểm tra bằng mắt (ảnh margin 1.25 không bị crop mất mặt).
- [ ] Resume hoạt động (kill giữa chừng, chạy lại tiếp được).
- [ ] Cell cuối dọn cache; `/kaggle/working` chỉ còn `ckpt/`, `results/`, `ff_common.py`, `train_ddp.py`.
- [ ] Cuối notebook có cell markdown tóm tắt: model tốt nhất, nâng cấp nào đáng tiền nhất (acc tăng bao nhiêu / tốn thêm thời gian bao nhiêu), các quyết định agent tự chọn.
