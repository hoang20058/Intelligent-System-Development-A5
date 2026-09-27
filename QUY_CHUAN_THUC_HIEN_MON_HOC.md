# HỆ THỐNG QUY CHUẨN & NGUYÊN TẮC BẮT BUỘC

## HỌC PHẦN: PHÁT TRIỂN HỆ THỐNG THÔNG MINH (INTELLIGENT SYSTEM DEVELOPMENT)

> **Mục đích**: Bộ tài liệu tiêu chuẩn kỹ thuật đóng vai trò "cẩm nang" áp dụng xuyên suốt cho các Bài tập lớn (Assignment 02, 03, 04...) và Đồ án môn học.

---

### 1. NGUYÊN TẮC PHẠM VI DỮ LIỆU & BẢO MẬT WORKSPACE

- **Giới hạn thư mục tuyệt đối**: Chỉ đọc, ghi và tương tác bên trong các thư mục được chỉ định rõ ràng cho bài tập (ví dụ: `A2_v1`, `A3`, `A4`...). Tuyệt đối không tự ý truy cập hoặc quét các thư mục môn học khác trong máy tính nếu không có yêu cầu.
- **Tách bạch cấu trúc nộp bài**: Tạo thư mục nộp bài chuẩn (ví dụ `A3_submit/`) phân tách rõ ràng từng bài toán (Phase 1, Phase 2, Phase 3...) cùng file báo cáo học thuật.

---

### 2. QUY TẮC CẤU TRÚC JUPYTER NOTEBOOK ("1 CELL = 1 NHIỆM VỤ/KẾT QUẢ ĐỘC LẬP")

- **Không gộp lệnh/kết quả**: Tuyệt đối không gộp nhiều tác vụ chạy ra kết quả vào chung 1 code cell.
  - Mối lệnh có thể ra kết quả yêu cầu làm 1 cell riêng
  - Mỗi hình vẽ đồ thị, ma trận nhầm lẫn hoặc bảng so sánh xuất ra trên **1 cell riêng**.
- **Cấu trúc xen kẽ Code - Markdown**: Cứ sau mỗi cell thực thi có kết quả, **bắt buộc có ngay 1 cell Markdown** đi kèm để giải thích:
  - Bản chất toán học / thuật toán vừa thực hiện.
  - Phân tích ý nghĩa của kết quả thực nghiệm vừa in ra.


### 5. NGUYÊN TẮC TRUNG THỰC DỮ LIỆU & ZERO HARDCODING

- **100% dữ liệu thực nghiệm**: Tuyệt đối không gán cứng (hardcode) giá trị số liệu trong bảng kết quả, 10 mẫu dự đoán ngẫu nhiên hoặc các nhận xét định lượng.
- **Tự động trích xuất**: Mọi con số trong bảng so sánh và báo cáo phải được lấy từ biến lưu trữ kết quả chạy thực tế của mô hình.

---

### 6. QUY CHUẨN BÁO CÁO HỌC THUẬT (LATEX, PDF & WORD .docx)

1. **Hình thức văn bản chuẩn Luận văn / Báo cáo**:
   - Định dạng hỗ trợ: Microsoft Word (.docx), LaTeX (.tex) và PDF (.pdf).
   - Font: **Times New Roman**, cỡ 12pt nội dung, 1.25 line spacing, lề chuẩn A4 (Trái 3cm, Trên/Dưới/Phải 2cm).
   - Bảng biểu có phối màu chuyên nghiệp (Header Navy #1B365D, dòng so le zebra striping).
   - Khung chú thích (Callout Boxes) với vạch viền nổi bật.
2. **Quy chuẩn Trang bìa tinh gọn**:
   - Trang 1 (Trang bìa): Tập trung đúng tiêu đề Báo cáo bài tập lớn, Tên đề tài nghiên cứu, kèm thông tin Sinh viên thực hiện và Mã sinh viên. Lược bỏ các phần tiêu ngữ/logo không bắt buộc.
   - Trang 2: Bỏ khung nộp bài rườm rà riêng biệt ở đầu, chuyển nội dung chuyên môn vào thẳng Mục lục và Chương 1.
3. **Đường link Dữ liệu Thực nghiệm (Kaggle)**:
   - Được nhúng trực tiếp ngay trước mỗi mục giới thiệu và phân tích dữ liệu ở Chương 2.
4. **Liên kết Mã nguồn (GitHub Repository)**:
   - Đặt trang trọng ở phần **Phụ lục (Appendix)** cuối cùng của tài liệu báo cáo.
5. **Nhúng trực quan hóa thực tế**:
   - Toàn bộ hình ảnh biểu đồ (Loss curves, Confusion matrices, Multi-metric bar charts, Master overview) phải được trích xuất trực tiếp từ các cell output của Jupyter Notebook và nhúng dưới dạng hình ảnh chất lượng cao kèm chú thích và phân tích kết quả chuyên sâu.
6. **Bám sát hệ thống câu hỏi lý thuyết môn học**:
   - Trả lời và phân tích sâu các khái niệm lý thuyết cốt lõi (hàm kích hoạt phi tuyến, công thức tính kích thước không gian đầu ra, lan truyền ngược qua Conv, code snippet minh họa).
   - Giải quyết trọn vẹn các bài tập tính toán tài nguyên (Bảng chi tiết Parameter Counting từng tầng nơ-ron).
   - Phân tích và biện luận hiện tượng quá khớp (Overfitting) trên dữ liệu kích thước nhỏ.
