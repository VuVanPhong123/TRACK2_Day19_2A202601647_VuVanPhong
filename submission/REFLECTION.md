# Reflection — Lab 19

**Tên:** Vũ Văn Phong  
**MSSV:** 2A202601647  
**Path đã chạy:** lite

## Câu hỏi (≤ 200 chữ)

Trên golden set, hybrid RRF thắng trung bình vì kết hợp tín hiệu lexical của BM25 với tín hiệu semantic của vector search. Query `exact` thường để BM25 mạnh nhất hoặc ngang hybrid vì thuật ngữ xuất hiện nguyên văn. Query `paraphrase` có lợi thế cho vector nếu embedding hiểu tốt tiếng Việt; trên lite path, `bge-small-en` khiến nhóm này yếu hơn. Query `mixed` là nơi hybrid ổn định nhất vì dùng được cả hai tín hiệu.

Tôi không dùng hybrid cho exact lookup theo mã, ID hoặc thuật ngữ cần khớp literal; BM25 đơn giản và rẻ hơn. Nếu corpus thiên về semantic matching và ít lexical overlap, pure vector có thể đủ tốt, giảm độ phức tạp và latency.

Điều bất ngờ nhất là đổi embedding model có thể thay đổi chất lượng tiếng Việt và số chiều vector, vì vậy phải re-index.

## Bonus challenge

Không thực hiện bonus vì đây là phần optional, không thuộc yêu cầu core + NB5–NB8.
