# Báo cáo Day 7 — Agentic RAG & Context Engineering

- **Học viên:** Đoàn Anh Hùng — **MSSV:** `2A202603880` — **Lớp:** `L3B`
- **Phạm vi nộp:** Phần cơ bản (`85` điểm), tiếp nối trực tiếp dữ liệu và vector store từ Day 6.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Kế thừa nguyên vẹn bộ dữ liệu pháp luật giao thông đường bộ Việt Nam từ Day 6 gồm 4 tệp PDF Công báo điện tử Chính phủ (`luat-36-2024-qh15-p1`, `luat-36-2024-qh15-p2`, `nd-168-2024-ndcp-p1`, `nd-168-2024-ndcp-p2`). Theo khoản 2 Điều 15 Luật Sở hữu trí tuệ, văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả.
  - Tổng cộng `211` trang vật lý $\rightarrow$ **`209` trang nội dung hợp lệ** (loại 2 trang trống cuối phần 1 của Luật và Nghị định) với **`627` chunks** theo chiến lược `B_paragraph_900` (`max_size=900`, `overlap=0`, trung bình `669.2` ký tự/chunk, trung vị `746`, `p75=831`), xuất ra [data/chunks.json](../../data/chunks.json) qua [src/export_day6.py](../../src/export_day6.py).
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** `PostgreSQL 16.16` + `pgvector 0.8.7` chạy qua Docker (`day6_student_starter-postgres-1`, cổng `127.0.0.1:55433`, DB `day6_lab`, bảng `rag_chunks`, lọc `strategy='B_paragraph_900'`), dùng khoảng cách cosine chính xác (`embedding <=> %s::vector`, similarity $= 1 - \text{distance}$).
  - **Embedding model:** `baai/bge-m3` qua OpenRouter API (`1024` chiều, client SDK `openai==3.51.0`), khớp hoàn toàn với vector đã ingest từ Day 6.
- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent chat model:** `grok-4.3` qua xAI OpenAI-compatible API (`https://api.x.ai/v1`, `temperature=0.0`, `reasoning_effort="none"` để giảm độ trễ gọi tool/JSON), theo dõi chi phí thực tế theo `cost_in_usd_ticks` (`1 USD = 1e10 ticks`).
  - **Môi trường & lệnh chạy:** Python `3.13.5` quản lý qua `uv` ([pyproject.toml](../../pyproject.toml)); các cấu hình mẫu lưu tại [.env.example](../../.env.example) (`.env` chứa API key được gitignore).
  - **Phân chia dev/test (`36` ca tổng cộng):**
    - **`dev` (`26` ca):** `23` câu có đáp án (`q01`–`q17`, `q25`–`q30`, kế thừa từ Day 6 và hiệu đính span chuẩn nằm trọn trong từng chunk) + `3` câu thiếu bằng chứng (`u_dev_01`–`u_dev_03`).
    - **`test` (`10` ca độc lập):** `7` câu có đáp án (`t01`–`t07`) xây mới hoàn toàn trên các trang PDF tách biệt khỏi `dev` (`luat-36-p1` trang `14`, `15`; `luat-36-p2` trang `4`, `29`; `nd-168-p2` trang `4`, `11`) + `3` câu thiếu bằng chứng (`u_test_01`–`u_test_03`). Tập `test` được giữ kín trong suốt quá trình thử nghiệm trên `dev`.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án ([data/questions.json](../../data/questions.json)):** mỗi câu có `expected_answer` và danh sách `gold` (`source_id`, `page`, `evidence` khớp chuỗi nguyên văn trong một chunk của đúng nguồn/trang, được kiểm chứng tự động bởi [starter/baseline.py](../../starter/baseline.py) và [tests/test_fixture.py](../../tests/test_fixture.py)).
  - **Câu thiếu bằng chứng ([data/agent_cases.json](../../data/agent_cases.json)):** chọn 6 câu gần chủ đề giao thông (`mức thu phí sử dụng đường bộ`, `chu kỳ kiểm định xe ô tô`, `lệ phí sát hạch/cấp đổi bằng`, `mức phí bảo hiểm TNDS bắt buộc xe máy`, `kích thước vạch kẻ đường QCVN`, `thời hạn giải quyết sang tên xe`) sau khi kiểm tra toàn bộ `627` chunks xác nhận `0` đoạn chứa các quy định này (`absence_reason` ghi rõ trong dữ liệu). Các câu thiếu bằng chứng không đưa vào mẫu số của retrieval metric.
  - **Tiêu chí chấm & tiêu chí chọn cấu hình từ `dev`:**
    - Retrieval: `Hit@1`, `Evidence Hit@3`, `Context Evidence Rate` (context cuối cùng chứa ít nhất một chunk đạt evidence hit), `MRR@3`, độ trễ trung vị (`ms`) và chi phí (`USD`).
    - Agent: `Tool use`, phân bố số lượt gọi search, `Grounded answer` (`0` = sai/thiếu/từ chối nhầm, `1` = đúng một phần hoặc sai nguồn văn bản, `2` = đúng và đủ), `Citation validity` (citation nằm trong context cuối **và** nội dung chunk thực sự hỗ trợ khẳng định kèm theo), `Answer citation coverage`, `Abstention`, `False abstention` tổng hợp bằng [starter/agent_metrics.py](../../starter/agent_metrics.py), cùng điểm chất lượng context (`0/1/2`) cho toàn bộ `7` câu `test` có đáp án.
    - **Tiêu chí khóa cấu hình trên `dev` trước khi chạy `test`:** ưu tiên cấu hình có số câu đạt `Grounded answer = 2` cao nhất, `Citation validity` đạt mức cao nhất với `0` câu trả lời sai sự thật/nhầm văn bản, và `Abstention` đúng `3/3`.

## Retrieval và thử nghiệm

Các thí nghiệm retrieval chạy cùng cutoff `top_k=3` và cùng ngân sách ngữ cảnh **`budget=2400` ký tự** (khoảng 3 đoạn đầy đủ của `B_paragraph_900`). Cấu hình `budget=800` mặc định của starter không được dùng trong bảng đối chứng chính vì độ dài trung bình mỗi chunk đã là `669.2` ký tự (`p75=831`), khiến `800` ký tự loại bỏ hầu hết chunk hạng #2–#3; bước LLM rerank (`LlmRerankRetriever`) cũng được loại khỏi pipeline chọn lọc để tránh đội chi phí và độ trễ chat ở tầng retrieval.

| Cấu hình | Phương pháp, top-k, filter, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline (`tfidf_b2400`) | TF-IDF (`tokenize_vi`), `top_k=3`, không filter, `priority`, `b=2400` | Mốc từ khóa cơ bản ở mức ngân sách đủ chứa `top_k=3` | dev | 12/23 (0.5217) | 15/23 (0.6522) | 14/23 (0.6087) | Median `3.73 ms`/câu / `0 USD` |
| Phương pháp 2 (`dense_b2400`) | Dense `pgvector` cosine (`baai/bge-m3`), `top_k=3`, không filter, `priority`, `b=2400` | Vector ngữ nghĩa Day 6 giúp khớp câu hỏi diễn đạt khác từ khóa | dev | 15/23 (0.6522) | 20/23 (0.8696) | 20/23 (0.8696) | Median `43.25 ms`/câu / embed chung `0.00000745 USD` (`35.55s` cho `23` query) |
| Phương pháp 3 (`bm25_b2400`) | Okapi BM25 (`k1=1.5, b=0.75`), `top_k=3`, không filter, `priority`, `b=2400` | Chuẩn hóa độ dài đoạn văn cải thiện từ khóa quy phạm so với TF-IDF | dev | 11/23 (0.4783) | 16/23 (0.6957) | 15/23 (0.6522) | Median `5.15 ms`/câu / `0 USD` |
| Cải thiện 1 (`bm25_ctx_b2400`) | Contextual-Heading BM25 (ghép tiêu đề `Điều` khi index), `top_k=3`, rule filter, `priority_heading`, `b=2400` | Giả thuyết chunk giữa Điều dài mất chủ thể xe nếu thiếu tiêu đề `Điều` | dev | 15/23 (0.6522) | 17/23 (0.7391) | 16/23 (0.6957) | Median `5.51 ms`/câu / `0 USD` |
| Cải thiện 2 (`hybrid_rrf_b2400`) | Hybrid RRF (`dense` top-20 + `bm25` top-20, `k=60`), `top_k=3`, không filter, `priority`, `b=2400` | Kết hợp thứ hạng dense + lexical tăng độ chính xác đầu bảng | dev | 18/23 (0.7826) | 21/23 (0.9130) | 20/23 (0.8696) | Median `49.43 ms`/câu / dùng lại vector query |
| Cấu hình chọn (`hybrid_rrf_filtered_b2400`) | Hybrid RRF + `infer_safe_document_type_filter`, `top_k=3`, `priority_heading`, `b=2400` | Lọc an toàn `luat` \| `nghi_dinh` khi tên văn bản rõ + kèm đầu mục Điều | dev | 19/23 (0.8261) | 21/23 (0.9130) | 20/23 (0.8696) | Median `47.42 ms`/câu / dùng lại vector query |
| Cấu hình đã chọn (`hybrid_rrf_filtered_b2400`) | Hybrid RRF + rule filter + `priority_heading`, `b=2400` | Đánh giá cấu hình khóa trên `test` (`7` câu có đáp án) | test | 4/7 (0.5714) | 6/7 (0.8571) | 5/7 (0.7143) | Median `45.26 ms`/câu / embed chung `0.00000256 USD` (`4.87s` cho `7` query) |
| Baseline Dense (`dense_b2400`) | Dense `pgvector`, `top_k=3`, không filter, `priority`, `b=2400` | Đối chứng Baseline Day 6 trên `test` | test | 4/7 (0.5714) | 6/7 (0.8571) | 5/7 (0.7143) | Median `41.17 ms`/câu / dùng lại vector query |
| Baseline TF-IDF (`tfidf_b2400`) | TF-IDF, `top_k=3`, không filter, `priority`, `b=2400` | Đối chứng từ khóa thuần trên `test` | test | 6/7 (0.8571) | 7/7 (1.0000) | 7/7 (1.0000) | Median `18.09 ms`/câu / `0 USD` |

*(Ghi chú bổ sung trên `dev`: `MRR@3` tăng đều từ `0.5797` ở `tfidf_b2400`/`bm25_b2400` lên `0.6884` ở `bm25_ctx_b2400`, `0.7464` ở `dense_b2400`, `0.8406` ở `hybrid_rrf_b2400` và `0.8623` ở `hybrid_rrf_filtered_b2400`. Trên `test` (`n=7`), các câu hỏi trích dẫn cụm thuật ngữ pháp lý khá nguyên văn nên `tfidf_b2400` đạt `7/7` Hit@3 và `bm25_b2400`/`bm25_ctx_b2400` đạt `5/7` Hit@1, `6/7` Hit@3, `0.7857` MRR@3.)*

- **Phân tích lỗi `dev` và tác động cải tiến:**
  1. **Lỗi thiếu bối cảnh Điều và nhầm loại văn bản (`q14`, `q29`, `q06` trên `dev`):**
     - Ở `q14` (*"Theo Luật Trật tự, an toàn giao thông đường bộ..."*), `dense_b2400` không có filter nên lấy đoạn thủ tục ở Nghị định (`nd-168-2024-ndcp-p2:p028:para:c001`) làm citation của `baseline_dense_b2400` sai nguồn văn bản được hỏi. Ở `c1_hybrid_b2400` khi tắt `include_heading_hint`, `q06` (`luat-36-2024-qh15-p2:p003:para:c001`) bị model tự suy sai thành `"Điều 22"` thay vì `"Điều 27"`, còn `q29` (bằng A1 cũ trước 2025) không nhận được đoạn chuyển tiếp `Điều 89` (`luat-36-2024-qh15-p2:p058:para:c001`) nên agent trích quy định mới ở `p030:para:c001` (`125 cm³`) và kết luận ngược rằng xe `150cc` không được chạy.
     - **Cách thử & kết quả:** Bật `infer_safe_document_type_filter` cùng `include_heading_hint=True` (tự động chèn block `(Đầu mục Điều liên quan)` từ `NeighborIndex.find_preceding_article_heading`, `max_lookback=60`) trong `hybrid_rrf_filtered_b2400` và `c2_agentic_tools_b3500`. Kết quả `Hit@1` trên `dev` tăng từ `18/23` lên `19/23`; trong agent `c2`, cả `q06` (`Điều 27`), `q14` (`luat-36-p2:p033:para:c002`), `q15` và `q29` (`Điều 89`: `50 cm³` đến dưới `175 cm³`) đều trả lời đúng `grade=2`.
  2. **Lỗi mất tên loại xe ở các đoạn con thuộc `Điều 6` / `Điều 7` dài (`q11`, `q13`, `q25` trên `dev`):**
     - Trong Nghị định 168, `Điều 6` (ô tô) bắt đầu từ trang `p009` còn điểm phạt vượt đèn đỏ nằm ở `p016:para:c000`; `Điều 7` (mô tô) bắt đầu từ trang `p018` còn hành vi leo vỉa hè/vượt đèn đỏ nằm ở `p022:para:c001`. Trong các đoạn con này chỉ ghi `"người điều khiển xe..."` mà không lặp lại `"xe ô tô"` hay `"xe mô tô"`, khiến cả `dense` lẫn `bm25` dễ kéo nhầm sang đoạn `Điều 10` (người đi bộ) hoặc `Điều 14`.
     - **Cách thử & kết quả:** Thử nghiệm `bm25_ctx_b2400` ghép tiêu đề `Điều` liền trước vào chỉ mục BM25 giúp `BM25` đơn lẻ trên `dev` tăng từ `11/23` lên `15/23` `Hit@1` (`MRR@3` từ `0.5797` lên `0.6884`), tuy nhiên khi vào agent `c2`, do `grok-4.3` thường chuyển sang `method="keyword"` kèm từ khóa `"Nghị định 168"` hoặc toán tử `"OR"` nên vẫn bỏ sót `q11`, `q12`, `q13`, `q25`, `q26` trên `dev`.
- **Lý do chọn cấu hình chốt trên `dev` (`hybrid_rrf_filtered_b2400` cho retrieval; `c2_agentic_tools_b3500` cho agent):**
  - `hybrid_rrf_filtered_b2400` đạt `Hit@1 = 19/23` (`0.8261`), `Hit@3 = 21/23` (`0.9130`), `MRR@3 = 0.8623` cao nhất trên `dev`.
  - `c2_agentic_tools_b3500` đạt nhiều câu mức `2` nhất trên `dev` (`17/23`), `Citation validity = 19/19` (`100%` citation đã kiểm tra được nguồn hỗ trợ trực tiếp), `Abstention = 3/3`, và `0` câu trả lời sai sự thật/sai nguồn văn bản.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Baseline (`baseline_dense_b2400`):** Dùng vòng lặp JSON trong [starter/agentic.py](../../starter/agentic.py) với adapter [BaselineOpenRouterController](../../src/agent.py), `PgVectorDenseRetriever`, `top_k=3`, `budget=2400`, `max_searches=4`. Dừng khi model trả `{"action": "answer", ...}`, hết `max_searches` (`search_limit`), hoặc citation không hợp lệ (`invalid_or_missing_citations`).
  - **Cải thiện 1 (`c1_hybrid_b2400`):** Dùng [run_improved_agent](../../src/agent.py) với OpenAI function calling (`search_documents` hỗ trợ `hybrid`/`dense`, `submit_final_answer`), `hybrid_rrf`, `top_k=3`, `budget=2400`, `max_searches=3`, `pack_context_priority` (`pinned_ids=context_ids`), `include_heading_hint=False`.
  - **Cải thiện 2 — cấu hình chốt (`c2_agentic_tools_b3500`):** Mở thêm tool `method="keyword"` (`bm25`) và `read_neighbor_chunk(chunk_id, direction)`, bật `use_safe_rule_filter=True` và `include_heading_hint=True`, nâng `budget=3500` (`max_searches=3`), truyền phần ngữ cảnh mới (`delta new_context`) ở mỗi lượt tool call để tránh nhân bản prompt token qua nhiều vòng.
  - **Xử lý lỗi & bảo vệ citation:** Lỗi HTTP/timeout hoặc lỗi DB PostgreSQL được truyền thẳng ra trace (`status="error"`, không silent fallback hay mock); hàm gửi kết quả `submit_final_answer` bắt buộc mọi `chunk_id` trong `citations` phải nằm trong `context_ids` hiện tại và xuất hiện nguyên văn dạng `[chunk-id]` trong `answer`, nếu vi phạm hệ thống chuyển sang `abstain`.

| Cấu hình / split | Tool use | Số lượt gọi (phân bố search) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_dense_b2400` / `dev` | 26/26 | Tổng `46` (`1`:19, `3`:1, `4`:6); TB `1.77` | `0`:5, `1`:2, `2`:16 (N=23) / `1.4783` | 17/21 (`80.95%`) | 18/18 | 3/3 | 5/23 |
| `c1_hybrid_b2400` / `dev` | 26/26 | Tổng `55` (`1`:9, `2`:5, `3`:12); TB `2.12` | `0`:9, `1`:1, `2`:13 (N=23) / `1.1739` | 18/20 (`90.00%`) | 15/15 | 3/3 | 8/23 |
| `c2_agentic_tools_b3500` / `dev` | 26/26 | Tổng `50` (`1`:13, `2`:2, `3`:11); `hybrid`:24, `bm25`:24, `neighbor`:2 | `0`:6, `1`:0, `2`:17 (N=23) / `1.4783` | 19/19 (`100.00%`) | 17/17 | 3/3 | 6/23 |
| `baseline_dense_b2400` / `test` | 10/10 | Tổng `15` (`1`:8, `3`:1, `4`:1); TB `1.50` | `0`:1, `1`:0, `2`:6 (N=7) / `1.7143` | 8/9 (`88.89%`) | 7/7 | 3/3 | 0/7 |
| `c2_agentic_tools_b3500` / `test` | 10/10 | Tổng `20` (`1`:5, `3`:5); `hybrid`:12, `bm25`:8 | `0`:1, `1`:0, `2`:6 (N=7) / `1.7143` | 8/8 (`100.00%`) | 6/6 | 3/3 | 1/7 |

- **Đánh giá chất lượng Context (`0/1/2`) cho toàn bộ `7` câu `test` có đáp án:**
  - **`baseline_dense_b2400` (`test`):** `0`: 1 (`t04`), `1`: 0, `2`: 6 (`t01`, `t02`, `t03`, `t05`, `t06`, `t07`) $\rightarrow$ trung bình **`12/7 = 1.71/2`**. Ở `t04`, dù dense retrieval tìm thấy `p004:para:c001` ở hạng 3, giới hạn `budget=2400` sau khi nạp `p004:c000` và `p008:c002` đã loại mất `p004:c001` khỏi context cuối.
  - **`c2_agentic_tools_b3500` (`test`):** `0`: 0, `1`: 0, `2`: 7 (`t01`–`t07`) $\rightarrow$ trung bình **`14/7 = 2.00/2`**. Nhờ `budget=3500` và `safe_rule_filter`, cả `p004:c000` và `p004:c001` đều nằm đủ trong context ở `t04`; toàn bộ `7/7` câu đều có evidence nguồn trong context cuối (riêng `t05` agent bỏ sót ý trong `nd-168-2024-ndcp-p2:p024:para:c002` dù chunk đã nằm trong context).

- **Chi tiết toàn bộ `10` ca `test` của cấu hình chốt `c2_agentic_tools_b3500` (kèm đối chiếu với `baseline_dense_b2400`):**

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation (`c2_agentic_tools_b3500`) | Nhãn, lý do (cùng đối chiếu `baseline_dense_b2400`) |
| :--- | :--- | :--- | :--- | :--- |
| `t01` | Có đáp án | 1 lượt `hybrid_rrf` (`flt=luat`) $\rightarrow$ top-1 `luat-36-p1:p014:para:c001` | Xe thô sơ đi làn phải trong cùng; xe cơ giới, xe máy chuyên dùng đi làn bên trái `[luat-36-2024-qh15-p1:p014:para:c001]` | Context `2`, Answer `2`, Cit `True` (`[True]`). Baseline cũng đạt `2`. |
| `t02` | Có đáp án | 1 lượt `hybrid_rrf` (`flt=None`) $\rightarrow$ hit `luat-36-p1:p014:para:c002` | Vượt bên trái; ngoại lệ vượt phải khi xe trước rẽ trái hoặc xe chuyên dùng làm việc không thể vượt trái `[luat-36-2024-qh15-p1:p014:para:c002]` | Context `2`, Answer `2`, Cit `True` (`[True]`). Baseline đạt `2`. |
| `t03` | Có đáp án (tình huống) | 2 lượt `hybrid_rrf` + 1 lượt `bm25` $\rightarrow$ `luat-36-p1:p019:para:c000`, `p015:para:c000` | Không được bóp còi lúc 23h trong phố; từ 22h–05h chỉ báo hiệu xin vượt bằng đèn `[luat-36-2024-qh15-p1:p019:para:c000]`, `[luat-36-2024-qh15-p1:p015:para:c000]` | Context `2`, Answer `2`, Cit `True` (`[True, True]`). Cả hai trích dẫn khớp Điều 21.1b và Điều 14.5 Luật 36. Baseline đạt `2`. |
| `t04` | Có đáp án (đa đoạn) | 1 lượt `hybrid_rrf` (`flt=luat`) $\rightarrow$ `p004:c000` + `p004:c001` | Nêu đủ 4 nhóm trường hợp a, b, c, d và yêu cầu an toàn `[luat-36-2024-qh15-p2:p004:para:c000]`, `[luat-36-2024-qh15-p2:p004:para:c001]` | Context `2`, Answer `2`, Cit `True` (`[True, True]`). **Baseline hỏng** (`Context=0`, `Answer=0`, `Cit=[False]`): rớt `p004:c001` vì hết `b=2400`, chỉ in `"trong trường hợp sau đây:"`. |
| `t05` | Có đáp án | 2 lượt `hybrid_rrf` + 1 lượt `bm25` $\rightarrow$ có `nd-168-p2:p024:para:c002` trong context | `"Không đủ bằng chứng để trả lời dựa trên tài liệu hiện có."` (`[]`) | Context `2`, **Answer `0` (`False abstention`)**: có `nd-168-p2:p024:para:c002` ghi *"giấy biên nhận... tổ chức tín dụng"* nhưng model tìm cụm *"giấy xác nhận"* ở Luật 36 nên từ chối. **Baseline đạt `2`** ở câu này. |
| `t06` | Có đáp án | 1 lượt `bm25` (`flt=nghi_dinh`) $\rightarrow$ `nd-168-p2:p004:para:c000` | Phạt tiền từ 40.000.000 đến 50.000.000 đồng với hành vi điểm a, b `[nd-168-2024-ndcp-p2:p004:para:c000]` | Context `2`, Answer `2`, Cit `True` (`[True]`). Baseline đạt `2`. |
| `t07` | Có đáp án | 1 lượt `bm25` (`flt=nghi_dinh`) $\rightarrow$ `nd-168-p2:p011:para:c000` | Trưởng CA cấp xã đến 2.500.000 đồng; Trưởng CA cấp huyện đến 15.000.000 đồng `[nd-168-2024-ndcp-p2:p011:para:c000]` | Context `2`, Answer `2`, Cit `True` (`[True]`). Baseline đạt `2`. |
| `u_test_01` | Thiếu bằng chứng | 3 lượt `bm25` (tìm phí bảo hiểm TNDS) $\rightarrow$ `submit_final_answer(abstain)` | `"Không đủ bằng chứng để trả lời dựa trên tài liệu hiện có."` (`[]`) | `Abstention = True`. Tài liệu chỉ nêu nghĩa vụ mang chứng nhận, không có biểu phí. Baseline dừng vì `search_limit` (`True`). |
| `u_test_02` | Thiếu bằng chứng | 2 lượt `hybrid_rrf` + 1 lượt `bm25` $\rightarrow$ `submit_final_answer(abstain)` | `"Không đủ bằng chứng để trả lời dựa trên tài liệu hiện có."` (`[]`) | `Abstention = True`. Không có kích thước cm của vạch kẻ đường. Baseline từ chối qua fallback `invalid_or_missing_citations` (`True`). |
| `u_test_03` | Thiếu bằng chứng | 3 lượt `hybrid_rrf` (tìm thời hạn sang tên) $\rightarrow$ `submit_final_answer(abstain)` | `"Không đủ bằng chứng để trả lời dựa trên tài liệu hiện có."` (`[]`) | `Abstention = True`. Không có số ngày làm việc thủ tục sang tên xe. Baseline từ chối qua fallback `invalid_or_missing_citations` (`True`). |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - **Retrieval summaries & traces:** [dev summary](../../results/retrieval-dev/summary.json), [test summary](../../results/retrieval-test/summary.json).
  - **Agent `dev` traces, nhãn `reviews.json` & metrics (`summary.json`):**
    - `baseline_dense_b2400`: [traces](../../results/agent-baseline_dense_b2400-dev/traces.json), [reviews](../../results/agent-baseline_dense_b2400-dev/reviews.json), [summary](../../results/agent-baseline_dense_b2400-dev/summary.json) — `142.29s`, `72` chat calls (`77,782` tok), `0.07873677 USD`.
    - `c1_hybrid_b2400`: [traces](../../results/agent-c1_hybrid_b2400-dev/traces.json), [reviews](../../results/agent-c1_hybrid_b2400-dev/reviews.json), [summary](../../results/agent-c1_hybrid_b2400-dev/summary.json) — `550.55s`, `79` chat calls (`147,939` tok), `0.1300049 USD`.
    - `c2_agentic_tools_b3500`: [traces](../../results/agent-c2_agentic_tools_b3500-dev/traces.json), [reviews](../../results/agent-c2_agentic_tools_b3500-dev/reviews.json), [summary](../../results/agent-c2_agentic_tools_b3500-dev/summary.json) — `378.82s`, `75` chat calls (`197,595` tok), `0.19304174 USD`.
  - **Agent `test` traces, nhãn `reviews.json` & metrics (`summary.json`):**
    - `baseline_dense_b2400`: [traces](../../results/agent-baseline_dense_b2400-test/traces.json), [reviews](../../results/agent-baseline_dense_b2400-test/reviews.json), [summary](../../results/agent-baseline_dense_b2400-test/summary.json) — `41.88s`, `25` chat calls (`26,059` tok), `0.02404627 USD`.
    - `c2_agentic_tools_b3500`: [traces](../../results/agent-c2_agentic_tools_b3500-test/traces.json), [reviews](../../results/agent-c2_agentic_tools_b3500-test/reviews.json), [summary](../../results/agent-c2_agentic_tools_b3500-test/summary.json) — `54.23s`, `30` chat calls (`76,446` tok), `0.06956526 USD`.
  - **Lệnh chạy tái lập:**
    - Unit test: `uv run python -m unittest discover -s tests -v` (`23/23` `OK` trong `0.239s`).
    - Retrieval: `uv run python -m src.eval_retrieval --split dev` và `--split test`.
    - Agent: `uv run python -m src.eval_agent --config <baseline_dense_b2400|c1_hybrid_b2400|c2_agentic_tools_b3500> --split <dev|test>`.
    - Tổng hợp nhãn: `python starter/agent_metrics.py --cases data/agent_cases.json --traces <traces> --reviews <reviews> --split <dev|test> --output <summary>`.
- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - **Trước/sau & đánh đổi:** Chuyển từ `baseline_dense_b2400` sang `c2_agentic_tools_b3500` nâng `Citation validity` từ `17/21` (`80.95%`) lên `19/19` (`100%`) trên `dev` và từ `8/9` (`88.89%`) lên `8/8` (`100%`) trên `test`, đồng thời nâng chất lượng context `test` từ `1.71/2` lên `2.00/2` (giải được ca đa đoạn `t04` mà `baseline` đánh rơi chunk) và tạo cơ chế từ chối chủ động qua `submit_final_answer(abstain)` thay vì dựa vào `search_limit` hay `invalid_or_missing_citations` như baseline (`u_test_01`–`u_test_03`). Đánh đổi là thời gian chạy và chi phí cao hơn (`378.82s` / `0.1930 USD` so với `142.29s` / `0.0787 USD` trên `dev`; `54.23s` / `0.0696 USD` so với `41.88s` / `0.0240 USD` trên `test`) và tỷ lệ từ chối thận trọng tăng nhẹ (`t05` trên `test`; `q12`, `q13`, `q27` trên `dev`).
  - **Giới hạn hiện tại:** (1) Tập `test` quy mô nhỏ (`10` ca = `7` có đáp án + `3` thiếu bằng chứng) nên sai lệch 1 ca tương ứng `14.3%`; (2) Với `reasoning_effort="none"` trên `grok-4.3`, model đôi khi chủ động chọn `method="keyword"` (`bm25`) ngay ở lượt đầu hoặc chèn ngoặc kép/`OR` vào query dù `Bm25Retriever` chỉ tách từ đơn giản; (3) Phần chunk ở `pgvector` giữ nguyên từ Day 6 (`B_paragraph_900` không kèm tiền tố `Điều` trong embedding dense) nên truy vấn nhắc tên xe vẫn có thể trượt các điểm con dài giữa `Điều 6`/`Điều 7`; (4) Toàn bộ nhãn đánh giá thủ công (`reviews.json`) và điểm context do tác giả trực tiếp đọc và chấm trên trace sau khi chạy xong.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện (`N/A` — bài nộp tập trung hoàn chỉnh phần Cơ bản `85` điểm trên bộ văn bản pháp luật giao thông `209` trang từ Day 6; tệp tài liệu lớn trước đó không dùng đã được xóa khỏi workspace).
