# AGENT GUIDE — Phân loại giới tính (nhị phân) trên FairFace với ResNet50 và các biến thể

> Tài liệu này dành cho coding agent. Hãy đọc toàn bộ trước khi viết code. Làm theo đúng thứ tự các bước, không bỏ bước, không tự ý đổi thiết kế trừ khi mục "Quy tắc" cho phép.

---

## 0. Mục tiêu

Xây dựng một dự án PyTorch hoàn chỉnh, tái lập được (reproducible), để:

1. Huấn luyện **mô hình gốc hoàn thiện nhất** (ResNet50 pretrained ImageNet, fine-tune đầy đủ, pipeline huấn luyện chuẩn) cho bài toán **phân loại giới tính nhị phân (Male/Female)** trên **FairFace**.
2. Tạo **3 biến thể chỉnh sửa từ mô hình gốc** (tổng 4 mô hình) và so sánh công bằng với nhau.
3. Xuất đầy đủ **số liệu, biểu đồ, bảng so sánh, phân tích lỗi và phân tích công bằng theo chủng tộc** để đưa vào báo cáo bài tập lớn.
4. Code phải **chạy được trên Google Colab, Kaggle Notebook và máy local** mà không cần sửa tay (xem mục 0.1).

Nguyên tắc so sánh: **chỉ thay đổi đúng một yếu tố kiến trúc giữa mô hình gốc và mỗi biến thể**; mọi thứ khác (dữ liệu, augmentation, optimizer, schedule, seed, số epoch) giữ giống hệt.

---

## 0.1 Môi trường chạy: Colab / Kaggle / Local (BẮT BUỘC hỗ trợ)

Mục tiêu: cùng một codebase, chỉ khác phần tự động nhận diện môi trường. Tạo `src/env.py` với hàm `detect_env()` trả về `"kaggle"`, `"colab"` hoặc `"local"`:

- Kaggle: tồn tại biến môi trường `KAGGLE_KERNEL_RUN_TYPE` hoặc thư mục `/kaggle/input`.
- Colab: import được `google.colab` hoặc tồn tại biến `COLAB_GPU`.
- Còn lại: local.

### Đường dẫn theo môi trường (không hard-code)

| Môi trường | Dữ liệu (đọc) | Kết quả/checkpoint (ghi) | Ghi chú |
|------------|---------------|--------------------------|---------|
| Kaggle | `/kaggle/input/<tên-dataset>/` (read-only) | `/kaggle/working/outputs/` | Add dataset FairFace vào notebook qua *Add Input*. Không ghi vào `/kaggle/input` |
| Colab | `/content/data/` hoặc Google Drive | `/content/drive/MyDrive/fairface_gender/outputs/` | Mount Drive để không mất checkpoint khi runtime bị ngắt |
| Local | `./data/` | `./outputs/` | Theo `configs/default.yaml` |

- Viết hàm `resolve_paths(cfg)` trong `env.py` trả về `data_dir`, `output_dir`, `splits_dir` theo bảng trên; cho phép ghi đè bằng tham số dòng lệnh `--data_dir`, `--output_dir`.
- **Tự dò dữ liệu**: nếu `data_dir` không rõ, quét `/kaggle/input` (hoặc thư mục dữ liệu) tìm file `train_labels.csv` / `val_labels.csv` (không phân biệt hoa thường, có thể nằm trong thư mục con, tên file có thể là `fairface_label_train.csv` / `fairface_label_val.csv`), rồi suy ra thư mục ảnh từ cột `file`. In đường dẫn tìm được ra log. Nếu không tìm thấy → dừng và in hướng dẫn cụ thể (xem bên dưới), không tự bịa dữ liệu.
- Kaggle ghi chỉ được vào `/kaggle/working` (giới hạn dung lượng khoảng 20 GB): **không copy cả dataset sang working**; chỉ lưu `splits/*.csv`, checkpoint và hình.

### Lấy dữ liệu FairFace
- **Kaggle**: dùng *Add Input* → tìm dataset FairFace (bản có ảnh padding 0.25 kèm file nhãn CSV). Notebook có thể để Internet tắt vẫn chạy được.
- **Colab**: một trong ba cách, chọn bằng `cfg.data_source`:
  1. `drive`: dữ liệu đã upload sẵn lên Google Drive → mount và đọc (nên **giải nén/copy về `/content/data` (đĩa local của Colab)** trước khi train vì đọc ảnh từ Drive rất chậm).
  2. `kaggle_api`: dùng `kaggle datasets download -d <slug> -p /content/data --unzip` (cần `kaggle.json` do người dùng tự upload; agent chỉ in hướng dẫn, không tự tạo khóa).
  3. `url`: tải từ link do người dùng cung cấp (`gdown` cho Google Drive).
- Agent phải hỏi/để trống chỗ `<slug>` và `<url>` cho người dùng điền; **không tự đoán tên dataset**.

### Ràng buộc tài nguyên (GPU miễn phí: Colab T4 ~15 GB, Kaggle T4 x2 hoặc P100 ~16 GB)
- Mặc định **dùng 1 GPU** (không bắt buộc DataParallel). Nếu Kaggle có 2 GPU, có thể bật tùy chọn `cfg.multi_gpu=true` (`nn.DataParallel`), mặc định tắt để đảm bảo tái lập.
- `num_workers`: tự chọn `min(4, os.cpu_count())` (Colab/Kaggle chỉ có khoảng 2–4 CPU). Nếu DataLoader treo hoặc lỗi shared memory → hạ về `num_workers=2` rồi `0`.
- Batch size 64 với AMP ở 224×224 chạy được trên T4 16 GB. Nếu `CUDA out of memory`: tự giảm một nửa batch và dùng gradient accumulation để giữ batch hiệu dụng 64 (đã có trong mục 3). Với M3 (CBAM) tốn thêm bộ nhớ, ưu tiên thử batch 32 × accum 2 nếu M3 bị OOM.
- Có tùy chọn `cfg.cache_images` (mặc định false): nếu RAM đủ, tiền resize ảnh 224×224 và lưu thành tensor uint8 `.npy`/`.pt` để các epoch sau đọc nhanh hơn; kiểm tra RAM trước khi bật.
- **Giới hạn thời gian phiên**: Colab miễn phí có thể ngắt sau vài giờ hoặc khi idle; Kaggle giới hạn mỗi phiên khoảng 9–12 giờ và có quota GPU hàng tuần. Vì vậy:
  - Lưu checkpoint `last.pt` **mỗi epoch** (gồm model, optimizer, scheduler, scaler AMP, epoch, best metric, RNG state, lịch sử log) và `best.pt` khi cải thiện.
  - `train.py` có cờ `--resume` (tự động resume nếu thấy `last.pt` trong thư mục output). Nếu mô hình đã train xong (có `DONE` flag file) thì **bỏ qua** để chạy lại notebook không train lại.
  - `run_all` chạy **tuần tự từng mô hình và độc lập** (mỗi mô hình một lệnh/cell riêng) để nếu phiên bị ngắt, chỉ cần chạy lại cell đó.
- Kaggle: nhớ bật GPU (Settings → Accelerator) và lưu notebook bằng *Save Version → Save & Run All* khi muốn giữ output; checkpoint cần nằm trong `/kaggle/working` để tải về/ tái sử dụng làm Dataset cho phiên sau.
- Cài thư viện: chỉ `pip install` những gói **chưa có** sẵn (torch, torchvision, numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, pyyaml đều có sẵn trên cả hai nền tảng). Nếu thiếu gói, code phải **degrade gracefully**:
  - `statsmodels` không có → tự cài McNemar bằng `scipy.stats.chi2`/`binomtest`.
  - `fvcore`/`thop` không có → bỏ qua FLOPs.
  - `wandb` mặc định tắt (dùng TensorBoard hoặc chỉ CSV log).
  - **Không pin phiên bản torch/torchvision** trong `requirements.txt` (để tránh phá môi trường có sẵn trên Colab/Kaggle); chỉ ghi phiên bản tối thiểu. Dùng cú pháp `weights=ResNet50_Weights.IMAGENET1K_V2` và có fallback `pretrained=True` nếu phiên bản torchvision cũ.
  - Tải trọng số pretrained cần Internet: nếu Kaggle tắt Internet, hướng dẫn người dùng bật Internet hoặc thêm Dataset chứa trọng số ResNet50 và đặt đường dẫn qua `cfg.pretrained_path`; code kiểm tra và báo lỗi rõ ràng, không âm thầm train từ đầu.

---

## 1. Bốn mô hình cần cài đặt

| ID | Tên | Thay đổi so với M0 | Mục đích |
|----|-----|--------------------|----------|
| **M0** | `resnet50_base` | Không (mô hình gốc) | Baseline hoàn thiện nhất |
| **M1** | `resnet50_head` | Thay đầu phân loại: GeM pooling + MLP head (2048→512→1) có BatchNorm + Dropout | Đánh giá ảnh hưởng của head/pooling |
| **M2** | `resnet50_se` | Chèn khối **SE (Squeeze-and-Excitation)** vào mỗi Bottleneck | Đánh giá attention theo kênh |
| **M3** | `resnet50_cbam` | Chèn khối **CBAM** (channel + spatial attention) vào mỗi Bottleneck | Đánh giá attention theo kênh + không gian |

### 1.1 M0 — `resnet50_base`
- `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`.
- Thay `fc` bằng `nn.Linear(2048, 1)` (một logit, dùng `BCEWithLogitsLoss`). Quy ước nhãn: **Male = 1, Female = 0**.
- Fine-tune toàn bộ mạng, dùng **learning rate phân tầng** (discriminative LR): backbone `lr_backbone`, head `lr_head = 10 × lr_backbone`.
- Dropout 0.2 trước `fc` (cố định cho M0, M2, M3).

### 1.2 M1 — `resnet50_head`
- Giữ nguyên backbone như M0.
- Thay `avgpool` bằng **GeM pooling** (p khởi tạo = 3, học được).
- Head: `Linear(2048,512) → BatchNorm1d(512) → ReLU → Dropout(0.3) → Linear(512,1)`.
- Đây là biến thể duy nhất thay đổi head; không thay đổi Bottleneck.

### 1.3 M2 — `resnet50_se`
- Viết module `SEBlock(channels, reduction=16)`: `GAP → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → nhân lại với feature map`.
- Chèn SE **sau `bn3` và trước phép cộng residual** trong mọi `Bottleneck` của ResNet50 (có thể bọc/patch `Bottleneck.forward`).
- Load trọng số pretrained cho phần backbone gốc (`strict=False`); các lớp SE khởi tạo mặc định. Với FC cuối của SE, khởi tạo bias sao cho đầu ra sigmoid ≈ 1 (ví dụ bias = 3) để mạng khởi đầu gần với ResNet50 gốc.
- Head giữ nguyên như M0.

### 1.4 M3 — `resnet50_cbam`
- Viết `ChannelAttention` (GAP + GMP → shared MLP, reduction=16, cộng rồi sigmoid) và `SpatialAttention` (mean + max theo kênh → conv 7×7 → sigmoid), ghép thành `CBAM`.
- Vị trí chèn và cách load pretrained **giống M2** (sau `bn3`, trước residual add; `strict=False`; khởi tạo để gần identity).
- Head giữ nguyên như M0.

> Kiểm tra bắt buộc cho M2, M3: in số tham số và số tham số tăng thêm so với M0; chạy forward với tensor `[2,3,224,224]` đảm bảo ra shape `[2,1]`.

---

## 2. Dữ liệu — FairFace

### 2.1 Thông tin dataset (agent phải kiểm tra lại trên máy thực tế)
- FairFace gồm ảnh khuôn mặt đã crop, nhãn trong CSV: `train_labels.csv`, `val_labels.csv` với các cột `file, age, gender, race, service_test`.
- `gender` nhận giá trị `Male` / `Female`; `race` gồm 7 nhóm (White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern).
- Có hai bản: padding 0.25 và padding 1.25. **Dùng bản padding 0.25** làm mặc định (đặt đường dẫn trong config, cho phép đổi).
- Tên file/thư mục nhãn có thể khác nhau tùy nguồn dataset (đặc biệt trên Kaggle): dùng cơ chế tự dò ở mục 0.1 và chuẩn hóa tên cột về `file, age, gender, race` trước khi dùng. Kiểm tra đường dẫn trong cột `file` (ví dụ `train/1.jpg`) khớp với vị trí ảnh thực tế; in 5 đường dẫn mẫu và xác nhận file tồn tại.
- Các thông tin trên (tên cột, kích thước ảnh, số lượng) cần được **xác minh bằng code** (in ra shape, giá trị unique, số mẫu); nếu khác thì dùng thực tế và ghi chú vào log.

### 2.2 Chia tập
- Tập `val` gốc của FairFace → dùng làm **TEST** (không đụng tới khi huấn luyện/chọn mô hình).
- Tập `train` gốc → chia **train / validation = 90/10**, stratified theo tổ hợp `gender × race`, `seed=42`.
- Lưu danh sách file của mỗi tập ra `splits/{train,val,test}.csv` để tái lập.
- Kiểm tra: không trùng file giữa 3 tập; in phân bố gender và race của từng tập.

### 2.3 Tiền xử lý & Augmentation
- **Train**: `Resize(224,224)` (ảnh đã crop sẵn nên có thể dùng `RandomResizedCrop(224, scale=(0.8,1.0))`) → `RandomHorizontalFlip(0.5)` → `RandomRotation(15)` → `ColorJitter(brightness=0.15, hue=0.04)` → `ToTensor` → `Normalize(ImageNet mean/std)`.
- **Val/Test**: `Resize(224,224)` → `ToTensor` → `Normalize`.
- Không dùng augmentation làm thay đổi giới tính (không dùng vá ghép kiểu CutMix/MixUp trong bản chính).
- `DataLoader`: `num_workers` cấu hình được, `pin_memory=True`, `persistent_workers=True` khi có workers.

---

## 3. Cấu hình huấn luyện (GIỐNG NHAU cho cả 4 mô hình)

| Tham số | Giá trị |
|---------|---------|
| Loss | `BCEWithLogitsLoss` (không dùng class weight; xử lý ở mục 3.1) |
| Optimizer | `AdamW`, `weight_decay=1e-4` |
| LR | `lr_backbone=1e-4`, `lr_head=1e-3` |
| Scheduler | Warmup tuyến tính 2 epoch → Cosine annealing |
| Epoch tối đa | 25 |
| Early stopping | patience 5 theo `val_loss` (đồng thời lưu checkpoint tốt nhất theo `val_acc`) |
| Batch size | 64 (tự giảm khi thiếu VRAM, ghi vào log; dùng gradient accumulation để giữ batch hiệu dụng = 64) |
| Checkpoint | `last.pt` mỗi epoch + `best.pt`; hỗ trợ `--resume` (xem mục 0.1) |
| Số epoch khi tài nguyên hạn chế | Cho phép `--epochs` ghi đè (ví dụ 15) nhưng phải **dùng cùng giá trị cho cả 4 mô hình** |
| Mixed precision | `torch.cuda.amp` (AMP) bật mặc định |
| Gradient clipping | `max_norm=1.0` |
| Seed | 42 (chính); nếu còn thời gian chạy thêm seed 1 và 2 |

### 3.1 Cân bằng lớp
- Tính tỷ lệ Male/Female trên train. Nếu lệch nhẹ (tỷ lệ lớp thiểu số > 40%) → không cần xử lý; nếu lệch hơn → dùng `pos_weight` trong `BCEWithLogitsLoss`. Ghi quyết định vào log.

### 3.2 Tái lập
- Hàm `set_seed(seed)` đặt seed cho `random`, `numpy`, `torch`, `torch.cuda`; `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- Lưu `config.yaml` kèm mỗi lần chạy.

---

## 4. Đánh giá

### 4.1 Chỉ số trên tập TEST (cho từng mô hình)
- Accuracy, Precision, Recall, F1 (macro và theo từng lớp), ROC-AUC, PR-AUC.
- Confusion matrix (số lượng và chuẩn hóa).
- Ngưỡng quyết định mặc định 0.5; **không** tối ưu ngưỡng trên test. Có thể tìm ngưỡng tốt nhất trên tập val và báo cáo thêm như một dòng riêng.

### 4.2 Phân tích công bằng (đặc trưng của FairFace — bắt buộc)
- Tính Accuracy, F1, và False Positive / False Negative Rate **theo từng nhóm `race`** (7 nhóm) và **theo nhóm `age`**.
- Xuất bảng và biểu đồ cột; chỉ ra nhóm tốt nhất, nhóm kém nhất, và **khoảng chênh lệch (max − min)** của accuracy giữa các nhóm.
- So sánh xem biến thể nào thu hẹp khoảng chênh lệch này.

### 4.3 Phân tích lỗi
- Lưu 16–32 ảnh bị dự đoán sai có độ tin cậy cao nhất của mỗi mô hình (lưới ảnh kèm nhãn thật/dự đoán/xác suất).
- Với M0: sinh **Grad-CAM** (lớp `layer4`) cho một số ảnh đúng và sai để minh họa vùng mà mô hình chú ý.

### 4.4 So sánh giữa các mô hình
- Bảng tổng hợp: số tham số, FLOPs (nếu có `fvcore`/`thop`, nếu không thì bỏ qua), thời gian huấn luyện/epoch, thời gian suy luận/ảnh, Accuracy, F1, AUC, chênh lệch công bằng theo race.
- Nếu chạy ≥3 seed: báo cáo mean ± std.
- Kiểm định ý nghĩa thống kê giữa M0 và từng biến thể: **McNemar test** trên dự đoán của tập test (dùng `statsmodels` hoặc tự cài).
- Biểu đồ: đường cong loss/accuracy train–val của 4 mô hình trên cùng một hình; ROC của 4 mô hình trên một hình.

---

## 5. Cấu trúc dự án

```
fairface_gender/
├── configs/
│   └── default.yaml
├── data/                      # dữ liệu FairFace (không commit)
├── splits/                    # train.csv, val.csv, test.csv
├── src/
│   ├── dataset.py             # FairFaceDataset, transforms, dataloader
│   ├── models/
│   │   ├── __init__.py        # build_model(name)
│   │   ├── base.py            # M0
│   │   ├── head_variant.py    # M1 (GeM + MLP head)
│   │   ├── se.py              # SEBlock + M2
│   │   └── cbam.py            # CBAM + M3
│   ├── train.py               # vòng lặp huấn luyện
│   ├── evaluate.py            # metric, fairness, confusion matrix, ROC
│   ├── gradcam.py
│   ├── compare.py             # tổng hợp 4 mô hình, McNemar, bảng/biểu đồ
│   ├── env.py                 # detect_env(), resolve_paths(), dò dữ liệu, kiểm tra GPU/RAM
│   └── utils.py               # seed, logger, checkpoint (save/resume), đếm tham số
├── notebooks/
│   └── run_colab_kaggle.ipynb # notebook chạy end-to-end, mỗi mô hình một cell
├── scripts/
│   ├── prepare_splits.py
│   └── run_all.sh             # chạy lần lượt M0→M3 rồi compare (local)
├── outputs/
│   └── {model_name}/{seed}/   # checkpoint, log, metrics.json, hình
├── requirements.txt
└── README.md
```

Giao diện dòng lệnh tối thiểu:

```bash
python scripts/prepare_splits.py --data_dir data/fairface --out splits   # --data_dir có thể bỏ trống để tự dò
python -m src.train --model resnet50_base --config configs/default.yaml --seed 42 --resume
python -m src.evaluate --model resnet50_base --seed 42
python -m src.compare --models resnet50_base resnet50_head resnet50_se resnet50_cbam
```

### Notebook `notebooks/run_colab_kaggle.ipynb` (các cell theo thứ tự)

1. **Môi trường**: in `detect_env()`, phiên bản torch/torchvision, `nvidia-smi`, RAM/disk; cảnh báo nếu chưa bật GPU.
2. **Code**: Colab → `git clone <repo>` hoặc giải nén zip mã nguồn đã upload; Kaggle → thêm mã nguồn qua Dataset/Input rồi `sys.path.append(...)`. Để `<repo_url>` cho người dùng điền.
3. **Drive (chỉ Colab)**: `drive.mount('/content/drive')`, đặt `output_dir` trên Drive.
4. **Dữ liệu**: lấy theo `cfg.data_source`, rồi chạy tự dò + `prepare_splits.py`; in thống kê.
5. **Kiểm tra nhanh**: smoke test 1–2 epoch trên tập con (cờ `--debug`) cho cả 4 mô hình.
6. **Huấn luyện**: 4 cell riêng cho M0, M1, M2, M3 (mỗi cell gọi `src.train ... --resume`).
7. **Đánh giá & so sánh**: chạy `src.evaluate` và `src.compare`, hiển thị bảng và hình ngay trong notebook.
8. **Đóng gói kết quả**: nén `outputs/` (không gồm `last.pt` nếu quá nặng) thành zip để tải về / lưu Drive.

Mọi cell phải **chạy lại an toàn** (idempotent): không train lại mô hình đã có `DONE`, không xóa dữ liệu hay checkpoint cũ.

---

## 6. Thứ tự thực hiện (agent làm tuần tự, kiểm tra sau mỗi bước)

1. **Khởi tạo**: tạo cấu trúc thư mục, `src/env.py` (mục 0.1), `requirements.txt` (chỉ ghi phiên bản tối thiểu, không pin torch/torchvision; các gói: numpy, pandas, scikit-learn, matplotlib, seaborn, pyyaml, tqdm, scipy, pillow; tùy chọn: statsmodels, tensorboard).
2. **Khám phá dữ liệu**: đọc CSV, in thống kê, vẽ phân bố gender/race/age; xác nhận thực tế khớp mục 2.1.
3. **Chia tập** (`prepare_splits.py`) và kiểm tra không rò rỉ.
4. **Dataset + DataLoader**: chạy thử 1 batch, lưu lưới ảnh minh họa sau augmentation để kiểm tra bằng mắt.
5. **Cài M0** và **overfit thử 1 batch nhỏ** (loss phải giảm gần 0) để xác nhận pipeline đúng.
6. **Hoàn thiện `train.py`** (AMP, scheduler, early stopping, checkpoint, log CSV từng epoch, hỗ trợ resume).
7. **Cài M1, M2, M3**; chạy kiểm tra shape và đếm tham số.
8. **Chạy thử nhanh** (1–2 epoch, tập con 2.000 ảnh) cho cả 4 mô hình để chắc chắn không lỗi.
9. **Huấn luyện đầy đủ** 4 mô hình (seed 42), mỗi mô hình chạy độc lập và có thể resume; thêm seed khác nếu còn thời gian/quota GPU.
10. **Đánh giá** từng mô hình (mục 4.1–4.3).
11. **So sánh** (mục 4.4) và xuất bảng/biểu đồ vào `outputs/compare/`.
12. **Tạo `notebooks/run_colab_kaggle.ipynb`** theo mục 0.1 và kiểm tra các cell chạy được tuần tự (ít nhất ở chế độ `--debug`).
13. **Viết `README.md`** có hướng dẫn riêng cho Colab, Kaggle và local, cách resume khi bị ngắt phiên; tạo `outputs/RESULTS.md` chứa các bảng kết quả dạng markdown để copy vào báo cáo.

---

## 7. Quy tắc cho agent

- **Không rò rỉ dữ liệu**: tập test chỉ dùng ở bước đánh giá cuối; mọi lựa chọn (checkpoint, ngưỡng, siêu tham số) dựa trên tập validation.
- **Không thay đổi siêu tham số riêng cho từng mô hình** để “làm đẹp” kết quả; nếu muốn tinh chỉnh, áp dụng cùng một thay đổi cho cả 4 mô hình và ghi lại.
- Mọi đường dẫn, siêu tham số nằm trong `configs/default.yaml`, không hard-code; đường dẫn dữ liệu/kết quả phải đi qua `resolve_paths()` để chạy được trên Colab, Kaggle và local.
- Không ghi vào thư mục chỉ đọc (`/kaggle/input`); không yêu cầu quyền root hay cài gói hệ thống (`apt`); không phụ thuộc đĩa bền vững của Colab (luôn lưu checkpoint lên Drive hoặc `/kaggle/working`).
- Không in/log khóa API (ví dụ `kaggle.json`); hướng dẫn người dùng tự cung cấp.
- Code có type hint cơ bản, docstring ngắn, logging rõ ràng (in cấu hình, số tham số, thời gian mỗi epoch).
- Nếu thiếu GPU/VRAM: giảm batch size + gradient accumulation, hoặc giảm số worker; **không** giảm độ phân giải xuống dưới 224 trừ khi báo rõ trong log.
- Nếu một bước thất bại (thiếu dữ liệu, lỗi phiên bản thư viện), dừng lại, báo nguyên nhân cụ thể và đề xuất cách sửa thay vì tự bịa dữ liệu.
- Mỗi hình vẽ phải có tiêu đề, nhãn trục, chú giải; lưu PNG độ phân giải ≥ 150 dpi.
- Kết quả cuối phải **đúng với số liệu thật đã chạy**; không điền số liệu giả vào báo cáo.

---

## 8. Danh sách kiểm tra hoàn thành (Definition of Done)

- [ ] `splits/*.csv` tồn tại, không trùng lặp giữa các tập.
- [ ] 4 mô hình huấn luyện xong, có checkpoint tốt nhất và `metrics.json`.
- [ ] Bảng so sánh tổng hợp (mục 4.4) và biểu đồ loss/acc, ROC.
- [ ] Phân tích công bằng theo race và age cho cả 4 mô hình.
- [ ] Confusion matrix, ảnh dự đoán sai, Grad-CAM của M0.
- [ ] McNemar test M0 vs M1/M2/M3.
- [ ] `README.md` và `outputs/RESULTS.md` hoàn chỉnh, chạy lại được bằng `scripts/run_all.sh` (local).
- [ ] `notebooks/run_colab_kaggle.ipynb` chạy được trên Colab và Kaggle (ít nhất đã smoke test với `--debug`); resume sau khi ngắt phiên hoạt động.
- [ ] Code tự nhận diện môi trường và tự dò dữ liệu; không còn đường dẫn hard-code.

---

## 9. Gợi ý nhận xét cho báo cáo (agent chỉ ghi nhận số liệu, người dùng tự viết kết luận)

- Biến thể attention (SE, CBAM) có cải thiện đáng kể so với M0 không, và đổi lại tăng bao nhiêu tham số/thời gian?
- Thay đổi head/pooling (M1) ảnh hưởng ra sao tới độ chính xác và độ ổn định giữa các seed?
- Nhóm race/age nào cho kết quả thấp nhất, và biến thể nào giảm khoảng chênh lệch?
- Các lỗi tập trung ở loại ảnh nào (trẻ em, ảnh mờ, góc nghiêng, đeo kính…)?