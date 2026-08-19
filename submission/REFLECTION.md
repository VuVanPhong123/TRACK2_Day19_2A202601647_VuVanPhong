# Reflection — Lab 19

**Tên:** Vũ Văn Phong  
**MSSV:** 2A202601647  
**Path đã chạy:** lite (NB2 override: `multilingual-lite`)

## Câu hỏi (≤ 200 chữ)

Trên golden set, hybrid RRF đạt 80.6%, cao hơn BM25 77.8% và vector 75.8%. Với `multilingual-lite`, exact hybrid đạt 98.0% (BM25 96.7%), paraphrase vector đạt 48.0% còn hybrid 44.0%, và mixed hybrid đạt 95.0% (BM25 97.0%). Điều này cho thấy hybrid cải thiện tổng thể nhưng vẫn có trade-off theo nhóm query; tôi giữ nguyên số liệu thay vì che giấu điểm BM25 còn tốt hơn ở mixed.

Tôi không dùng hybrid cho exact lookup theo mã, ID hoặc thuật ngữ cần khớp literal; BM25 đơn giản và rẻ hơn. Nếu corpus thiên về semantic matching và ít lexical overlap, pure vector có thể đủ tốt, giảm độ phức tạp và latency. RRF dùng `1/(60 + rank)` với rank bắt đầu từ 1.

Điều bất ngờ nhất là đổi embedding model có thể thay đổi chất lượng tiếng Việt và số chiều vector, vì vậy phải re-index. `multilingual-lite` được chọn cho NB2 vì có chất lượng paraphrase tốt hơn bge-small mà vẫn nhẹ hơn multilingual e5-large.

## Bonus challenge

Không thực hiện bonus vì đây là phần optional, không thuộc yêu cầu core + NB5–NB8.
