# 📋 KẾ HOẠCH TRIỂN KHAI A5 — CNN (v2, đã cập nhật)

---

## 📦 TỔNG QUAN DỮ LIỆU

| # | Dataset | Loại | Input Shape | Quy mô sử dụng |
|---|---------|------|-------------|-----------------|
| 1 | **CIFAR-10** | Ảnh màu, 10 classes | `32×32×3` | ~50k train + 10k test |
| 2 | **Cats vs Dogs** | Ảnh, binary | **`64×64×3`** (resize) | ~25k ảnh |
| 3 | **Diabetes (Playground S5E12)** | Tabular, binary | `(25, 1)` cho Conv1D | **250,000 dòng** (sample từ 700k) |

---

## 🏗️ CẤU TRÚC NỘP BÀI (4 items)

```
A5/
├── DATA/
│   ├── cifar10/
│   ├── Cat and dog/
│   └── playground-series-s5e12/
├── TaiLieu/
│
├── A5_Phase2_CIFAR10_CNN.ipynb       # Keras + PyTorch trên CIFAR-10
├── A5_Phase3_CatDog_CNN.ipynb        # Keras + PyTorch trên Cats vs Dogs
├── A5_Phase4_Diabetes_1DCNN.ipynb    # Keras + PyTorch trên Diabetes
│
└── A5_BaoCao.docx                    # Lý thuyết CNN + Tổng hợp so sánh
```

> [!IMPORTANT]
> **Lý thuyết CNN** (function composition, convolution, activation, pooling, BN, softmax, linear collapse, bài tập tính params) và **bảng tổng hợp so sánh TF vs PyTorch** → đưa vào **báo cáo Word**.

---

## 📐 KIẾN TRÚC CNN ĐÃ SỬA (fix MaxPool)

### 🖼️ CIFAR-10 (32×32) — Tối đa 2 MaxPool

**3-Layer CNN:**
```
Input (32×32×3)
  → Conv(32, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 16×16×32
  → Conv(64, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 8×8×64
  → Conv(128, 3×3, same) → BN → ReLU                    # → 8×8×128
  → GlobalAveragePooling2D                                # → 128
  → Dense(128) → ReLU → Dropout(0.5)
  → Dense(10, softmax)
```

**5-Layer CNN:**
```
Input (32×32×3)
  → Conv(32, 3×3, same) → BN → ReLU                     # → 32×32×32
  → Conv(64, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 16×16×64
  → Conv(128, 3×3, same) → BN → ReLU                    # → 16×16×128
  → Conv(256, 3×3, same) → BN → ReLU → MaxPool(2×2)    # → 8×8×256
  → Conv(512, 3×3, same) → BN → ReLU                    # → 8×8×512
  → GlobalAveragePooling2D                                # → 512
  → Dense(256) → ReLU → Dropout(0.5)
  → Dense(10, softmax)
```

> [!NOTE]
> Dùng `GlobalAveragePooling2D` thay `Flatten` → giảm params đáng kể, tránh overfitting trên CIFAR-10 nhỏ.

---

### 🐱🐶 Cats vs Dogs (64×64) — Tối đa 3 MaxPool

**3-Layer CNN:**
```
Input (64×64×3)
  → Conv(32, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 32×32×32
  → Conv(64, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 16×16×64
  → Conv(128, 3×3, same) → BN → ReLU → MaxPool(2×2)    # → 8×8×128
  → GlobalAveragePooling2D                                # → 128
  → Dense(64) → ReLU → Dropout(0.5)
  → Dense(1, sigmoid)                                     # Binary
```

**5-Layer CNN:**
```
Input (64×64×3)
  → Conv(32, 3×3, same) → BN → ReLU                     # → 64×64×32
  → Conv(64, 3×3, same) → BN → ReLU → MaxPool(2×2)     # → 32×32×64
  → Conv(128, 3×3, same) → BN → ReLU → MaxPool(2×2)    # → 16×16×128
  → Conv(256, 3×3, same) → BN → ReLU → MaxPool(2×2)    # → 8×8×256
  → Conv(512, 3×3, same) → BN → ReLU                    # → 8×8×512
  → GlobalAveragePooling2D                                # → 512
  → Dense(128) → ReLU → Dropout(0.5)
  → Dense(1, sigmoid)
```

---

### 📊 Diabetes 1D-CNN (25 features → Conv1D)

**3-Layer 1D-CNN:**
```
Input (25, 1)
  → Conv1D(64, 3, same) → BN → ReLU → MaxPool1D(2)     # → 12×64
  → Conv1D(128, 3, same) → BN → ReLU                    # → 12×128
  → Conv1D(256, 3, same) → BN → ReLU                    # → 12×256
  → GlobalAveragePooling1D                                # → 256
  → Dense(128) → ReLU → Dropout(0.5)
  → Dense(1, sigmoid)
```

**5-Layer 1D-CNN:**
```
Input (25, 1)
  → Conv1D(32, 3, same) → BN → ReLU                     # → 25×32
  → Conv1D(64, 3, same) → BN → ReLU → MaxPool1D(2)     # → 12×64
  → Conv1D(128, 3, same) → BN → ReLU                    # → 12×128
  → Conv1D(256, 3, same) → BN → ReLU → MaxPool1D(2)    # → 6×256
  → Conv1D(512, 3, same) → BN → ReLU                    # → 6×512
  → GlobalAveragePooling1D                                # → 512
  → Dense(256) → ReLU → Dropout(0.5)
  → Dense(1, sigmoid)
```

> [!TIP]
> Diabetes: `batch_size=512`, sample **250,000 dòng** từ 700k gốc. Đủ lớn để đại diện, tránh train quá lâu trên CPU.

---

## 📓 NOTEBOOK 1: `A5_Phase2_CIFAR10_CNN.ipynb`

### Cấu trúc cell-by-cell:

| # | Loại | Nội dung |
|---|------|---------|
| 1 | **MD** | Tiêu đề: "CIFAR-10 — CNN Classification (Keras & PyTorch)" |
| 2 | Code | Import thư viện (tensorflow, torch, sklearn, matplotlib, numpy...) |
| 3 | **MD** | Giải thích cấu hình môi trường |

#### PHẦN A: DATA PREPROCESSING (dùng chung cho cả 2 framework)

| # | Loại | Nội dung |
|---|------|---------|
| 4 | Code | Load ảnh từ `DATA/cifar10/` bằng `os.listdir` + `cv2.imread`, gán label |
| 5 | **MD** | Giải thích cách load, cấu trúc thư mục |
| 6 | Code | Hiển thị sample ảnh mỗi lớp (1 subplot grid) |
| 7 | **MD** | Mô tả dataset: 10 classes, kích thước, phân phối |
| 8 | Code | Thống kê phân phối class (bar chart) |
| 9 | **MD** | Nhận xét cân bằng/không cân bằng |
| 10 | Code | Normalize `[0,1]`, train/val split `80/20`, one-hot encode |
| 11 | **MD** | Giải thích preprocessing |

#### PHẦN B: KERAS — 3-Layer CNN

| # | Loại | Nội dung |
|---|------|---------|
| 12 | Code | Build model 3L Keras (`Sequential`), `model.summary()` |
| 13 | **MD** | Giải thích kiến trúc, đếm params |
| 14 | Code | Compile + Train (epochs=30, EarlyStopping, ReduceLR) — lưu `history` + đo `time` |
| 15 | **MD** | Nhận xét quá trình training |
| 16 | Code | Plot Training/Val Loss curves |
| 17 | **MD** | Phân tích overfitting/underfitting |
| 18 | Code | Plot Training/Val Accuracy curves |
| 19 | **MD** | Phân tích convergence |
| 20 | Code | Evaluate: tính Accuracy, Precision, Recall, F1 trên test set |
| 21 | **MD** | Phân tích hiệu năng |
| 22 | Code | Confusion Matrix heatmap |
| 23 | **MD** | Phân tích confusion: lớp nào dễ nhầm |

#### PHẦN C: KERAS — 5-Layer CNN

| # | Loại | Nội dung |
|---|------|---------|
| 24–35 | | Flow tương tự Phần B (12 cells: build → train → curves → eval → CM → MD) |

#### PHẦN D: KERAS — SO SÁNH 3L vs 5L

| # | Loại | Nội dung |
|---|------|---------|
| 36 | Code | Bảng so sánh metrics (từ biến, ZERO hardcoding) |
| 37 | **MD** | Phân tích bảng |
| 38 | Code | Grouped Bar Chart: Acc, Prec, Recall, F1 |
| 39 | **MD** | Nhận xét ảnh hưởng của depth |

#### PHẦN E: PYTORCH — 3-Layer CNN

| # | Loại | Nội dung |
|---|------|---------|
| 40 | Code | Chuyển dữ liệu sang torch tensor, tạo `DataLoader` |
| 41 | **MD** | Giải thích DataLoader |
| 42 | Code | Define `class CNN3Layer(nn.Module)` + print model + count params |
| 43 | **MD** | Giải thích kiến trúc |
| 44 | Code | Training loop tường minh (forward→loss→backward→step), đo time |
| 45 | **MD** | Giải thích training loop PyTorch vs Keras |
| 46 | Code | Plot Loss curves |
| 47 | **MD** | Phân tích |
| 48 | Code | Plot Accuracy curves |
| 49 | **MD** | Phân tích |
| 50 | Code | Evaluate: Acc, Prec, Recall, F1 |
| 51 | **MD** | Phân tích |
| 52 | Code | Confusion Matrix |
| 53 | **MD** | Phân tích |

#### PHẦN F: PYTORCH — 5-Layer CNN

| # | Loại | Nội dung |
|---|------|---------|
| 54–67 | | Flow tương tự Phần E |

#### PHẦN G: PYTORCH — SO SÁNH 3L vs 5L

| # | Loại | Nội dung |
|---|------|---------|
| 68–71 | | Bảng + Chart so sánh PyTorch internal |

#### PHẦN H: ĐỐI ĐẦU KERAS vs PYTORCH

| # | Loại | Nội dung |
|---|------|---------|
| 72 | Code | Bảng tổng hợp 4 mô hình: {3L,5L} × {Keras,PyTorch} — #params, time, metrics |
| 73 | **MD** | Phân tích tổng hợp |
| 74 | Code | Grouped Bar Chart đối đầu framework |
| 75 | **MD** | Kết luận cho CIFAR-10 |

---

## 📓 NOTEBOOK 2: `A5_Phase3_CatDog_CNN.ipynb`

**Cấu trúc giống hệt Notebook 1**, thay đổi:

| Khác biệt | Chi tiết |
|-----------|---------|
| Input shape | `64×64×3` (resize từ ảnh gốc) |
| Load data | `cv2.imread` + `cv2.resize(img, (64,64))` từ `DATA/Cat and dog/train/cats/` và `dogs/` |
| Output | `Dense(1, sigmoid)` — binary classification |
| Loss | `binary_crossentropy` (Keras) / `nn.BCEWithLogitsLoss` (PyTorch) |
| MaxPool | 3 lần (64→32→16→8) cho 3-Layer; 3 lần cho 5-Layer |
| Metrics | Thêm AUC-ROC nếu muốn (optional) |

---

## 📓 NOTEBOOK 3: `A5_Phase4_Diabetes_1DCNN.ipynb`

**Cấu trúc tương tự**, thay đổi:

| Khác biệt | Chi tiết |
|-----------|---------|
| Load data | `pd.read_csv('train.csv')`, sample 250k dòng: `df.sample(250000, random_state=42)` |
| EDA thêm | Correlation heatmap, phân phối target, xử lý missing |
| Encode | `LabelEncoder` / `pd.get_dummies` cho 6 cột categorical |
| Scale | `StandardScaler` trên features số |
| Reshape | `X.reshape(-1, 25, 1)` cho Conv1D |
| Conv layer | `Conv1D` (Keras) / `nn.Conv1d` (PyTorch) |
| batch_size | **512** |
| Output | `Dense(1, sigmoid)` — binary |

### Lưu ý đặc biệt cho Conv1D PyTorch:
```python
# PyTorch Conv1d expects input shape: (batch, channels, length)
# → reshape: (N, 1, 25) — khác Keras (N, 25, 1)
```

---

## 📄 BÁO CÁO WORD: `A5_BaoCao.docx`

### Cấu trúc báo cáo:

| Chương | Nội dung |
|--------|---------|
| **Trang bìa** | Thông tin nhóm, môn học |
| **Khung nộp bài** | Link GitHub, link Notebook, link Data source |
| **Chương 1: Lý thuyết CNN** | |
| 1.1 | Neuron → Hàm phi tuyến → Layer → Function Composition |
| 1.2 | Chứng minh Linear Collapse (toán học) |
| 1.3 | Phép tích chập 2D: công thức, weight sharing, output shape |
| 1.4 | Hàm kích hoạt: ReLU, LeakyReLU, GELU (công thức + đồ thị) |
| 1.5 | Pooling: MaxPool, AvgPool, Strided Conv |
| 1.6 | Batch Normalization (công thức) |
| 1.7 | Flatten + Dense + Softmax |
| 1.8 | Bài tập tính #params, memory, FLOPs (Slide 66) |
| 1.9 | Trả lời Self-Study Questions (Slide 54) |
| **Chương 2: Kết quả CIFAR-10** | Nhúng hình từ Notebook, bảng metrics, phân tích |
| **Chương 3: Kết quả Cats vs Dogs** | Tương tự |
| **Chương 4: Kết quả Diabetes** | Tương tự |
| **Chương 5: Tổng hợp so sánh** | Bảng master 12 mô hình, biểu đồ tổng hợp, kết luận |

---

## ⚙️ CẤU HÌNH TRAINING CHUẨN (Anaconda CPU-friendly)

| Tham số | CIFAR-10 | Cats vs Dogs | Diabetes |
|---------|----------|-------------|----------|
| Input | 32×32×3 | 64×64×3 | (25, 1) |
| Batch size | 64 | 64 | **512** |
| Epochs | 30 | 30 | 20 |
| Optimizer | Adam(lr=1e-3) | Adam(lr=1e-3) | Adam(lr=1e-3) |
| EarlyStopping | patience=5 | patience=5 | patience=3 |
| ReduceLR | patience=3, factor=0.5 | patience=3, factor=0.5 | patience=2, factor=0.5 |
| Data Augment | Flip, Rotation(15°) | Flip, Rotation(20°), Zoom | Không |
| Sample size | Full | Full | **250,000** |

---

## ✅ CHECKLIST TUÂN THỦ

| Quy chuẩn | Trạng thái |
|-----------|-----------|
| 1 Cell = 1 Nhiệm vụ | ✅ Mỗi plot, mỗi eval = cell riêng |
| Code ↔ Markdown xen kẽ | ✅ Sau mỗi cell output → MD giải thích |
| Zero Hardcoding | ✅ Mọi metrics từ biến |
| Không trùng A4 | ✅ CIFAR-10, CatDog, Playground S5E12 |
| MaxPool hợp lý | ✅ CIFAR-10: 2 Pool, CatDog: 3 Pool |
| Diabetes tối ưu | ✅ Sample 250k, batch=512 |
| CatDog resize | ✅ 64×64×3 |
| Anaconda CPU | ✅ batch/epoch phù hợp |

---

## 🔄 THỨ TỰ TRIỂN KHAI

```mermaid
graph LR
    A["Notebook 1\nCIFAR-10"] --> B["Notebook 2\nCats vs Dogs"]
    B --> C["Notebook 3\nDiabetes"]
    C --> D["Báo cáo Word\n(Lý thuyết + Tổng hợp)"]

    style A fill:#7B68EE,color:#fff
    style B fill:#7B68EE,color:#fff
    style C fill:#7B68EE,color:#fff
    style D fill:#E74C3C,color:#fff
```

| Bước | Nội dung | Thời gian ước tính |
|------|---------|-------------------|
| Notebook 1 | CIFAR-10: Preprocess → Keras 3L/5L → PyTorch 3L/5L → So sánh | ~4–5 giờ |
| Notebook 2 | CatDog: Preprocess → Keras 3L/5L → PyTorch 3L/5L → So sánh | ~4–5 giờ |
| Notebook 3 | Diabetes: Preprocess → Keras 3L/5L → PyTorch 3L/5L → So sánh | ~3–4 giờ |
| Báo cáo | Lý thuyết CNN + Nhúng hình + Tổng hợp | ~3–4 giờ |
| **Tổng** | | **~14–18 giờ** |
