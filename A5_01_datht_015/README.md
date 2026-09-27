# ASSIGNMENT 05: PHÂN TÍCH VÀ XÂY DỰNG MẠNG NƠ-RON TÍCH CHẬP (CNN)
## Thực nghiệm Đa tập dữ liệu: CIFAR-10, Cats vs Dogs và Diabetes Tabular

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Keras](https://img.shields.io/badge/Keras-TensorFlow-D00000.svg)](https://keras.io/)
[![LaTeX](https://img.shields.io/badge/LaTeX-XeLaTeX-008080.svg)](https://www.latex-project.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717.svg)](https://github.com/hoang20058/Intelligent-System-Development-A5)

---

## 📌 THÔNG TIN SINH VIÊN VÀ BÀI NỘP
* **Học phần:** Thiết kế và Phát triển Hệ thống Thông minh
* **Bài tập lớn:** Assignment 05 (A5)
* **Họ và tên sinh viên:** Hoàng Tiến Đạt
* **Mã sinh viên:** B23DCCE015
* **Mã thư mục nộp bài:** `A5_01_datht_015`
* **Kho lưu trữ GitHub:** [https://github.com/hoang20058/Intelligent-System-Development-A5](https://github.com/hoang20058/Intelligent-System-Development-A5)

---

## 📖 TỔNG QUAN ĐỀ TÀI & MỤC TIÊU NGHIÊN CỨU

Đề tài tập trung nghiên cứu bản chất toán học của **Mạng Nơ-ron Tích chập (Convolutional Neural Networks - CNN)** dưới góc nhìn **Hợp hàm Sâu (Function Composition)**, đồng thời tiến hành thực nghiệm toàn diện trên 3 miền bài toán với 3 cấu trúc dữ liệu khác biệt:
1. **Phân loại ảnh màu đa lớp tự nhiên (CIFAR-10):** Đánh giá năng lực trích xuất đặc trưng phân cấp trên 10 lớp đối tượng.
2. **Phân loại ảnh nhị phân mẫu nhỏ (Cats vs Dogs Small Dataset):** Nghiên cứu hiện tượng Overfitting, tính hữu dụng của Data Augmentation và Dropout.
3. **Phân loại dữ liệu bảng y tế bằng 1D-CNN (Diabetes Tabular 1D-CNN):** Mở rộng kỹ thuật tích chập sang dữ liệu phi không gian (100.000 hồ sơ bệnh nhân).

Thực nghiệm được đối chuẩn song song giữa **2 Framework học sâu hàng đầu (Keras/TensorFlow vs PyTorch)** và **2 biến thể độ sâu (Kiến trúc 3-Layer vs 5-Layer)**, nhằm rút ra các kết luận khoa học có giá trị thực tiễn cao.

---

## 🗂 CẤU TRÚC THƯ MỤC NỘP BÀI (`A5_01_datht_015`)

Thư mục chính thức chứa đầy đủ các sản phẩm bàn giao theo đúng quy chuẩn học thuật:

```text
A5_01_datht_015/
├── A5_BaoCao.pdf                  # Báo cáo toàn văn định dạng PDF (Biên dịch trực tiếp từ XeLaTeX)
├── A5_BaoCao.docx                 # Báo cáo toàn văn định dạng Word (Times New Roman 12pt, lề chuẩn luận văn)
├── A5_BaoCao.tex                  # Toàn bộ mã nguồn LaTeX học thuật chuẩn mực
├── A5_Phase2_CIFAR10_CNN.ipynb    # Jupyter Notebook thực nghiệm Chuyên đề 2 (CIFAR-10)
├── A5_Phase3_CatDog_CNN.ipynb     # Jupyter Notebook thực nghiệm Chuyên đề 3 (Cats vs Dogs)
├── A5_Phase4_Diabetes_1DCNN.ipynb # Jupyter Notebook thực nghiệm Chuyên đề 4 (Diabetes 1D-CNN)
└── README.md                      # Tài liệu thuyết minh hướng dẫn tái hiện và tổng kết
```

---

## 🔗 NGUỒN TẬP DỮ LIỆU THỰC NGHIỆM (KAGGLE LINKS)

Các tập dữ liệu được thu thập và tiền xử lý từ các nguồn chuẩn hóa trên Kaggle:

1. **Cats vs Dogs Image Classification:**
   * URL: [https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification](https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification)
   * Đặc điểm: Tập con chuẩn hóa kích thước 64x64, chia sẵn tập Train (557 ảnh) và Test (140 ảnh).
2. **CIFAR-10 Dataset:**
   * URL: [https://www.kaggle.com/datasets/ayush1220/cifar10](https://www.kaggle.com/datasets/ayush1220/cifar10)
   * Đặc điểm: 60.000 ảnh màu RGB 32x32 thuộc 10 lớp vật thể tự nhiên (50k train / 10k test).
3. **Diabetes Health Indicators Tabular Dataset:**
   * URL: [https://www.kaggle.com/competitions/playground-series-s5e12/data?select=train.csv](https://www.kaggle.com/competitions/playground-series-s5e12/data?select=train.csv)
   * Đặc điểm: Dữ liệu y tế 100.000 dòng, 16 thuộc tính sinh học, biến đổi thành dạng tensor 1D $(1 \times 16 \times 1)$ cho mô hình 1D-CNN.

---

## 📊 BẢNG TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN TOÀN DIỆN (MASTER BENCHMARK)

Bảng đối chuẩn tổng hợp hiệu năng giữa Keras và PyTorch qua 2 kiến trúc 3-Layer và 5-Layer trên cả 3 tập dữ liệu:

| Tập dữ liệu | Framework | Kiến trúc | Số tham số | Thời gian huấn luyện (s) | Test Loss | Test Accuracy (%) | F1-Score Macro (%) | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CIFAR-10** | **Keras** | 3-Layer CNN | 608,970 | 362.4s | 1.1842 | 57.86% | 56.42% | - |
| (Ảnh màu 10 lớp) | **Keras** | 5-Layer CNN | 1,250,000 | 833.1s | 0.7654 | **74.42%** | **73.80%** | - |
| | **PyTorch** | 3-Layer CNN | 608,970 | 48.2s | 1.1215 | 60.14% | 58.75% | - |
| | **PyTorch** | 5-Layer CNN | 1,250,000 | 115.8s | 0.7210 | **75.12%** | **74.65%** | - |
| **Cats vs Dogs** | **Keras** | 3-Layer CNN | 545,858 | 18.5s | 0.6120 | 67.86% | 66.50% | 0.7340 |
| (Ảnh nhỏ nhị phân) | **Keras** | 5-Layer CNN | 1,180,000 | 32.1s | 0.7420 | 62.14% | 61.20% | 0.6820 |
| | **PyTorch** | 3-Layer CNN | 545,858 | 8.4s | 0.5890 | **69.29%** | **68.15%** | **0.7510** |
| | **PyTorch** | 5-Layer CNN | 1,180,000 | 14.2s | 0.7100 | 63.57% | 62.40% | 0.6950 |
| **Diabetes Tabular** | **Keras** | 3-Layer 1D-CNN | 42,114 | 145.2s | 0.3421 | 86.45% | 76.82% | 0.8245 |
| (Bảng y tế 100k) | **Keras** | 5-Layer 1D-CNN | 98,434 | 210.8s | 0.3395 | 86.62% | 77.05% | 0.8270 |
| | **PyTorch** | 3-Layer 1D-CNN | 42,114 | 62.5s | 0.3410 | 86.51% | 76.90% | 0.8252 |
| | **PyTorch** | 5-Layer 1D-CNN | 98,434 | 92.1s | 0.3382 | **86.74%** | **77.08%** | **0.8285** |

---

## 🔬 KẾT LUẬN KHOA HỌC THEN CHỐT

1. **Về độ sâu mô hình (Model Depth):**
   * **Quy mô dữ liệu lớn, tính biến thiên cao (CIFAR-10):** Kiến trúc 5-Layer vượt trội hoàn toàn kiến trúc 3-Layer (+16% Accuracy, +17% F1-Score). Việc tăng độ sâu mở rộng trường tiếp nhận (Receptive Field) và dung lượng biểu diễn giúp mô hình phân tách tốt các lớp tương đồng.
   * **Dữ liệu kích thước nhỏ (Cats vs Dogs ~700 ảnh):** Kiến trúc 3-Layer là lựa chọn tối ưu Pareto. Kiến trúc 5-Layer gây sụt giảm nghiêm trọng hiệu năng kiểm thử (Accuracy giảm từ 69.29% xuống 63.57%) do hiện tượng quá khớp (Overfitting) vì số tham số gấp gần 2000 lần số mẫu huấn luyện.
   * **Dữ liệu bảng 1D (Diabetes Tabular):** Mạng 1D-CNN thể hiện tính ổn định cao, đạt xấp xỉ 86.7% Accuracy và F1-Score 77.08%, chứng minh tính khả thi của phép tích chập 1 chiều trong việc phát hiện tương quan cục bộ giữa các chỉ số y sinh học.

2. **Về sự tương đồng và sai khác giữa Keras và PyTorch:**
   * **Số lượng tham số:** Tương đồng tuyệt đối (chênh lệch dưới 0.1% do cách xử lý BatchNorm running mean/variance).
   * **Tốc độ thực thi:** PyTorch tận dụng tăng tốc phần cứng GPU CUDA với overhead nhỏ hơn Keras, cho tốc độ huấn luyện nhanh gấp từ **2.5 đến 7.2 lần** trên cùng phần cứng.

---

## 🚀 HƯỚNG DẪN TÁI HIỆN THỰC NGHIỆM (REPRODUCIBILITY)

### 1. Yêu cầu môi trường
* Python 3.10 trở lên
* CUDA Toolkit (nếu có GPU)
* Các thư viện cần thiết:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn python-docx
```

### 2. Thực thi thực nghiệm
Khởi động Jupyter Notebook và mở từng file notebook tương ứng để chạy trực tiếp:
* CIFAR-10: Mở và chạy `A5_Phase2_CIFAR10_CNN.ipynb`
* Cats vs Dogs: Mở và chạy `A5_Phase3_CatDog_CNN.ipynb`
* Diabetes Tabular: Mở và chạy `A5_Phase4_Diabetes_1DCNN.ipynb`

### 3. Tái tạo Báo cáo PDF & Word
Để tự động biên dịch lại toàn bộ tài liệu báo cáo:
```bash
python -X utf8 build_full_report.py
```
Hệ thống sẽ tự động xuất bản `A5_BaoCao.pdf` (qua XeLaTeX) và `A5_BaoCao.docx` (qua python-docx) vào thư mục `A5_01_datht_015/`.
