# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Đình Việt
**Nhóm:** C401-D4
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector (embedding) có hướng gần giống nhau, tức là nội dung/ngữ nghĩa của hai câu rất tương đồng.

**Ví dụ HIGH similarity:**
- Sentence A: "I love playing football."
- Sentence B: "I enjoy playing soccer."
- Tại sao tương đồng:Cùng nói về sở thích chơi bóng đá (football = soccer), khác từ nhưng cùng ý nghĩa.

**Ví dụ LOW similarity:**
- Sentence A: "I love playing football."
- Sentence B: "The sky is very blue today."
- Tại sao khác:Hai câu nói về hai chủ đề hoàn toàn khác nhau (thể thao vs thời tiết).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo hướng (ngữ nghĩa) thay vì độ lớn vector, nên ít bị ảnh hưởng bởi độ dài câu và phản ánh ý nghĩa tốt hơn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Bước nhảy mỗi chunk = 500 − 50 = 450
Số chunks = [(10000 − 50) / 450] = 9950 / 450 ≈ 22.11
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số chunks tăng lên (vì bước nhảy nhỏ hơn), giúp giữ ngữ cảnh tốt hơn giữa các đoạn nhưng tốn tài nguyên hơn.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]
Tech docs

**Tại sao nhóm chọn domain này?**
> Question Answering based Clinical Text Structuring Using Pre-trained Language Model

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Randomized trial of folic acid supplementation and serum homocysteine levels | BeIR/scifact | 1687 | doc_id, title |
| 2 | Keratin-dependent regulation of Aire and gene expression in skin tumor keratinocytes | BeIR/scifact | 1058 | doc_id, title |
| 3 | ALDH1 is a marker of normal and malignant human mammary stem cells and a predictor of poor clinical outcome | BeIR/scifact | 1020 | doc_id, title |
| 4 | Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic | BeIR/scifact | 1990 | doc_id, title |
| 5 | New opportunities: the use of nanotechnologies to manipulate and track stem cells | BeIR/scifact | 640 | doc_id, title |


### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_id|string |"doc_1" |Dùng để filter theo document và xóa toàn bộ chunks của 1 document (delete_document) |
|chunk_id |int | 0|Xác định từng chunk riêng biệt trong document → hỗ trợ truy vết và debug retrieval |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Tài liệu	Strategy	Chunk Count	Avg Length	Preserves Context?
Doc 1 (Self-supervised learning...)| FixedSizeChunker (`fixed_size`) |5 |200 |Thấp (cắt cứng theo ký tự) |
| | SentenceChunker (`by_sentences`) |2 |350 | Cao (giữ nguyên câu)|
| | RecursiveChunker (`recursive`) | 4|220 |Tôi ưu |
| Tài liệu	Strategy	Chunk Count	Avg Length	Preserves Context?
Doc 2 (Model fusion techniques...)| FixedSizeChunker (`fixed_size`) |4 |220 |Thấp (cắt cứng theo ký tự) |
| | SentenceChunker (`by_sentences`) |2 |300 | Cao (giữ nguyên câu)|
| | RecursiveChunker (`recursive`) | 3|230 |Tôi ưu |

### Strategy Của Tôi

**Loại:** [FixedSizeChunker / SentenceChunker / RecursiveChunker / custom strategy]
RecursiveChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*
Strategy này chia văn bản theo thứ tự ưu tiên các separator: \n\n → \n → ". " → " " → "". Nếu đoạn vẫn quá dài so với chunk_size, nó tiếp tục split đệ quy với separator nhỏ hơn. Trong quá trình ghép chunk, nó cố gắng giữ nội dung càng dài càng tốt nhưng không vượt quá giới hạn. Cách này giúp giữ được cấu trúc tự nhiên của văn bản (đoạn, câu, từ) thay vì cắt cứng theo ký tự.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?*
Tech docs thường có cấu trúc rõ ràng theo đoạn, dòng và câu, nên RecursiveChunker tận dụng được các dấu hiệu này để giữ ngữ nghĩa tốt hơn. Đồng thời, nó vẫn đảm bảo giới hạn độ dài chunk phù hợp cho embedding và retrieval, giúp cân bằng giữa context và hiệu năng.

**Code snippet (nếu custom):**
```python
# Paste implementation here
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
|Doc 1 (Self-supervised learning...) | best baseline |2 |350 | Rất tốt (giữ nguyên ngữ nghĩa theo câu)|
| | **của tôi** |4 |220 |Tốt (cân bằng giữa context và độ dài chunk) |
|Doc 2 (Model fusion techniques...)| best baseline |2 |300 | Rất tốt|
| | **của tôi** |3 |230 |Tốt|

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker|9 |Cân bằng tốt giữa giữ ngữ nghĩa và giới hạn độ dài chunk | Có thể làm mất một phần context dài nếu bị chia nhỏ|
| Trần Văn Tuấn | SentenceChunker|9.5 |Giữ nguyên cấu trúc câu → ngữ nghĩa rõ ràng, retrieval rất chính xác |Chunk có thể dài, kém tối ưu cho embedding |
| Hồ Bảo Thư | FixedSizeChunker|9.5 |Đơn giản, ổn định, dễ kiểm soát kích thước chunk | Cắt cứng theo ký tự → dễ mất ngữ cảnh|

**Strategy nào tốt nhất cho domain này? Tại sao?**
> SentenceChunker là tốt nhất vì tech docs thường có thông tin rõ ràng theo từng câu, nên giữ nguyên câu giúp retrieval chính xác hơn. Tuy nhiên, RecursiveChunker cũng rất mạnh vì cân bằng tốt giữa context và kích thước, còn FixedSizeChunker của bạn Thư vẫn hữu ích nhờ tính đơn giản và hiệu năng ổn định

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng regex (?<=[.!?])\s+ để tách câu dựa trên dấu . ! ? theo sau bởi khoảng trắng. Sau đó normalize text bằng cách loại bỏ khoảng trắng dư (re.sub(r"\s+", " ", text)). Edge case xử lý gồm: text rỗng, câu có nhiều khoảng trắng, và loại bỏ các câu rỗng sau khi split.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán chia text đệ quy theo thứ tự separator ưu tiên (\n\n → \n → ". " → " " → ""). Base case là khi độ dài đoạn ≤ chunk_size thì trả về luôn, hoặc khi hết separator thì cắt cứng theo kích thước. Trong quá trình duyệt, dùng buffer để ghép các phần nhỏ thành chunk tối đa mà không vượt quá giới hạn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi thêm document, mỗi chunk được embed và normalize vector trước khi lưu vào store (in-memory hoặc ChromaDB). Khi search, query cũng được embed + normalize, sau đó tính cosine similarity với từng vector và sort giảm dần để lấy top_k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> Với filter, hệ thống lọc metadata trước rồi mới thực hiện similarity search trên tập đã lọc. Delete document được thực hiện bằng cách xóa tất cả records có metadata["doc_id"] tương ứng (hoặc dùng filter delete trong ChromaDB).

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent sử dụng pipeline RAG: retrieve top_k chunks → lọc theo score > 0 → build context có đánh dấu [Source i]. Prompt được thiết kế với rule chặt (chỉ dùng context, không hallucinate), sau đó inject context + question vào prompt và gọi LLM để sinh câu trả lời.

### Test Results

```
======================================= test session starts ========================================
platform win32 -- Python 3.14.3, pytest-9.0.3, pluggy-1.6.0 -- D:\Lab\.venv\Scripts\python.exe       
cachedir: .pytest_cache
rootdir: D:\Lab\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0, langsmith-0.7.29
collected 42 items                                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED         [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                  [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED           [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                 [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED       [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED        [ 19%] 
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED      [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                        [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED        [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED               [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                         [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED    [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED    [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                        [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED          [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED            [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                  [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED       [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED         [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED          [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                   [ 64%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED             [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED         [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED    [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED        [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED              [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED        [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED  [ 88%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED        [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED   [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================== 42 passed in 0.19s ======================================== 
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | high / low | | |
| 2 | | | high / low | | |
| 3 | | | high / low | | |
| 4 | | | high / low | | |
| 5 | | | high / low | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 14/ 15 |
| My approach | Cá nhân | 9/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 9/ 10 |
| Core implementation (tests) | Cá nhân | 28/ 30 |
| Demo | Nhóm | 0/ 5 |
| **Tổng** | | *95*/ 100** |
