# Reflection — Lab 19

**Tên:** Vũ Văn Phong  
**MSSV:** 2A202601647  
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

Trên golden set, hybrid RRF thắng trung bình vì kết hợp được tín hiệu lexical của BM25 và tín hiệu semantic của vector search. Với query `exact`, BM25 thường mạnh nhất hoặc ngang hybrid vì từ khóa kỹ thuật xuất hiện trực tiếp trong corpus. Với `paraphrase`, vector search có lợi thế khi embedding model hiểu tốt ngôn ngữ; riêng lite path dùng `bge-small-en`, nên paraphrase tiếng Việt có thể yếu hơn mong đợi. Với `mixed`, hybrid ổn định nhất vì tận dụng được cả hai loại tín hiệu.

Tôi không dùng hybrid khi bài toán là exact lookup theo mã/ID/thuật ngữ cần khớp literal, khi đó BM25 đơn giản và rẻ hơn. Ngược lại, nếu corpus thiên về semantic matching và lexical overlap thấp, pure vector có thể đủ tốt, giảm độ phức tạp và latency so với chạy hai retriever rồi fusion.

## Điều ngạc nhiên nhất khi làm lab này

Chất lượng retrieval phụ thuộc mạnh vào embedding model; đổi model có thể thay đổi cả chất lượng tiếng Việt lẫn số chiều vector, nên phải re-index.

## Bonus challenge

Không thực hiện bonus vì đây là phần optional, không thuộc yêu cầu core + NB5–NB8.
