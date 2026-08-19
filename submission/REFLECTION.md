# Reflection — Lab 19

**Tên:** Vũ Văn Phong  
**MSSV:** 2A202601647  
**Path đã chạy:** lite; NB2 dùng `multilingual-mpnet` (768d).

## Câu hỏi (≤ 200 chữ)

Trên golden set, hybrid RRF đạt 81.0%, cao hơn BM25 77.8% và vector 80.6%. Với `multilingual-mpnet`, BM25 thắng ở exact (96.7% so với hybrid 94.0%), vector thắng ở paraphrase (55.3%), còn hybrid thắng ở mixed (97.5% so với BM25 97.0%). Overall hybrid thắng cả hai baseline, nhưng lợi thế phụ thuộc loại query.

Tôi không dùng hybrid cho exact lookup theo mã, ID hoặc thuật ngữ cần khớp literal; BM25 đơn giản và rẻ hơn. Nếu corpus thiên về semantic matching và ít lexical overlap, pure vector có thể đủ tốt, giảm độ phức tạp và latency. RRF dùng `1/(60 + rank)` với rank bắt đầu từ 1.

Candidate pool bất đối xứng (BM25 75, vector 15) giữ độ phủ lexical cho exact/mixed trong khi tránh để semantic long tail lấn át RRF. Đổi embedding model cũng đổi số chiều vector, vì vậy phải re-index.

## Bonus challenge

Không thực hiện bonus vì đây là phần optional, không thuộc yêu cầu core + NB5–NB8.
