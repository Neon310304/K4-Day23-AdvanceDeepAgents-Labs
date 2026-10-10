# Đối chiếu bài nộp với RUBRIC

Bản tự đánh giá sau khi chạy hệ thống thật trên Colab với 9router và Daytona.
Điểm tự chấm: **100/100 theo các tiêu chí bên dưới**. Đây là đánh giá nội bộ; điểm chính thức do giảng viên quyết định, đặc biệt phần chất lượng tổng hợp và kiểm tra trích dẫn ngẫu nhiên.

## Kết quả chạy

| Chủ đề | Nguồn | Task thật | Họ nguồn |
|---|---:|---:|---|
| [survey about world model](reports/survey-about-world-model.md) | 15 | 5 | arxiv, hf-search, web |
| [survey about reinforcement learning for LLM reasoning](reports/survey-about-reinforcement-learning-for-llm-reasoning.md) | 13 | 5 | arxiv, hf-search, web |
| [survey about LLM agents and tool use](reports/survey-about-llm-agents-and-tool-use.md) | 12 | 6 | arxiv, hf-daily, hf-search, web |
| [survey about video and multimodal generation](reports/survey-about-video-and-multimodal-generation.md) | 14 | 7 | arxiv, hf-search, web |
| [survey about efficient inference and small language models](reports/survey-about-efficient-inference-and-small-language-models.md) | 11 | 6 | arxiv, hf-search, web |

Mỗi chủ đề có đủ `.md`, `.sources.json`, `.meta.json`. Báo cáo và registry được agent viết/sửa trong sandbox, chạy finalizer gốc và validator ở đó rồi tải về. Bản nộp được đối chiếu SHA-256 với bản xuất từ Colab và từng tệp trên Drive; xem `REPORTS.sha256`. Không sửa tay báo cáo trên host.

Metadata giữ các lượt task/tool/token thật. `revisions` ghi riêng model, thời gian và usage của các lượt sửa thành công. Token chỉ gồm lead và lượt sửa, chưa gồm token của subagent; thời gian không tính các lần chạy thất bại bị loại.

## Bảng tự chấm

| Mục RUBRIC | Điểm tự chấm | Bằng chứng |
|---|---:|---|
| 1. Công cụ nguồn và retry | 20/20 | Năm công cụ gọi thật trên Colab; retry lỗi tạm thời, Retry-After, backoff/jitter/cap; xử lý Exa HTTP 200 rate limit; throttle arXiv 3 giây dùng chung giữa các luồng. Các nhánh lỗi được kiểm thử. |
| 2. Agent và uỷ quyền | 20/20 | Mọi meta có ≥3 task và ≥3 họ nguồn đúng URL; middleware gom ba đề xuất task thật rồi dispatch song song. Prompt mang đủ ngữ cảnh, định dạng notes và quy tắc dữ liệu web không đáng tin. Hạn mức model/tool riêng cho mọi vai trò. |
| 3. Sandbox | 10/10 | Dùng open_sandbox; upload/download được đối chiếu bytes, finalizer/validator chạy trong sandbox. Không tải khóa lên sandbox. Lỗi không ghi đè bản hợp lệ; các test xác nhận dọn dẹp và rollback. |
| 4. Trích dẫn | 15/15 | Cả năm báo cáo qua check_citations; 25 mẫu dưới đây được đối chiếu nguồn thật. Mẫu kiểm tra là bằng chứng cho các câu được chọn, không phải xác minh thực nghiệm độc lập toàn bộ báo cáo. |
| 5. Chất lượng năm báo cáo | 25/25 | Mỗi báo cáo có đủ TL;DR, Background, 3–6 phần theo chủ đề, Trends and open problems, References; tổng hợp so sánh hướng tiếp cận, nguồn nền tảng và nguồn gần đây. Các số liệu đã rà soát giữ baseline, phạm vi và đơn vị. |
| 6. Mã và repo | 10/10 | Cài đặt/chạy/đọc reports có trong README; requirements cố định phiên bản; bốn tệp cấp sẵn không đổi; public files và lịch sử Git đã quét khóa. |
| Tổng | **100/100** | Tự đánh giá, không phải điểm từ giảng viên. |

## Kiểm tra đã chạy

- `python -m unittest discover -s tests -v`: **81/81 PASS**, trên Windows và Colab. Có test chạy ba researcher thật qua graph Deep Agents với model thử nghiệm, xác nhận thực thi đồng thời; test repair đi hết ngân sách model mà không lỗi recursion.
- `python tools.py`: cả năm công cụ nguồn gọi mạng thật đạt trên Colab. Tool-calling probe với model đang cấu hình đạt.
- `python self_check.py`: đủ năm báo cáo, metadata, trích dẫn và Git/secrets đều OK ở repo nộp.
- `python research.py --all --resume`: kiểm tra bỏ qua cả năm báo cáo hợp lệ; không gọi LLM lại cho chúng.
- `git diff --check`: đạt; nội dung `model.py`, `sandbox.py`, `self_check.py`, `finalize_citations.py` khớp bản gốc. Guard ở notebook chỉ chuẩn hóa CRLF/LF khi so sánh, không sửa các tệp này.

## 25 mẫu đối chiếu nội dung

Nguồn gốc được kiểm tra qua abstract/trang công bố chính thức hoặc nội dung bài. Các kết quả benchmark là kết quả tác giả công bố, chưa được tái lập thực nghiệm. URL kiểm tra có thể là phiên bản primary của cùng bài được liệt kê trên Hugging Face.

### World Models: From Latent Dynamics and Imagination to Interactive Simulation and Reliable Control

| Mẫu | Nội dung đã đối chiếu | Nguồn bằng chứng |
|---:|---|---|
| 1 | PlaNet uses deterministic and stochastic dynamics and latent overshooting. | [Nguồn](https://arxiv.org/abs/1811.04551) |
| 2 | Genie 3 announcement reports 24fps, 720p and consistency for a few minutes. | [Nguồn](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) |
| 3 | ACID checks action consistency and reports results across four world models and six tasks. | [Nguồn](https://huggingface.co/papers/2607.02403) |
| 4 | Physics benchmark: 40 tasks, eight models, 1280 videos and best score57.76/100. | [Nguồn](https://arxiv.org/abs/2610.08791) |
| 5 | 100x ego-centric video scaling improves agent and object fidelity unevenly. | [Nguồn](https://arxiv.org/abs/2610.12464) |

### Reinforcement Learning for LLM Reasoning: Rewards, Optimization, Capability, and Evaluation

| Mẫu | Nội dung đã đối chiếu | Nguồn bằng chứng |
|---:|---|---|
| 1 | Process supervision supplies intermediate-step feedback, unlike final-outcome feedback. | [Nguồn](https://arxiv.org/abs/2305.20050) |
| 2 | In the critical RLVR study, bases catch up at large k despite RL wins at small k. | [Nguồn](https://huggingface.co/papers/2504.13837) |
| 3 | ProRL uses KL control, reference-policy resetting and diverse tasks. | [Nguồn](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html) |
| 4 | Curriculum RL abstract reports average pass@256 gains of9.8 and10.3percentage points and teacher guidance. | [Nguồn](https://arxiv.org/abs/2606.22317) |
| 5 | DAPO reports50points onAIME2024 withQwen2.5-32B and code built onverl. | [Nguồn](https://huggingface.co/papers/2503.14476) |

### LLM Agents and Tool Use: Architectures, Orchestration, and Emerging Paradigms

| Mẫu | Nội dung đã đối chiếu | Nguồn bằng chứng |
|---:|---|---|
| 1 | ReAct kết hợp suy luận và hành động xen kẽ. | [Nguồn](https://arxiv.org/abs/2210.03629) |
| 2 | Toolformer học tự giám sát khi nào và cách gọi API. | [Nguồn](https://arxiv.org/abs/2302.04761) |
| 3 | Tool-to-Agent Retrieval dùng không gian vector chung; mức tăng Recall@5 và nDCG@5 là kết quả do tác giả báo cáo. | [Nguồn](https://huggingface.co/papers/2511.01854) |
| 4 | Benchmark chín framework: overhead có thể vượt 60 lần; mức giảm accuracy của cấu hình schema là tới 32 points trong abstract v2, không phải 32% tương đối. | [Nguồn](https://huggingface.co/papers/2602.03128) |
| 5 | Opera lưu ghi chú sửa lỗi và dùng đánh giá định kỳ hoặc theo sự kiện. | [Nguồn](https://huggingface.co/papers/2609.33987) |

### Survey on Video and Multimodal Generation: Foundations, Latent Dynamics, and Open Problems

| Mẫu | Nội dung đã đối chiếu | Nguồn bằng chứng |
|---:|---|---|
| 1 | DDPM đặt nền tảng tạo ảnh bằng diffusion và liên hệ huấn luyện với denoising score matching. | [Nguồn](https://arxiv.org/abs/2006.11239) |
| 2 | LDM chạy diffusion trong latent space của autoencoder đã huấn luyện để giảm chi phí so với pixel space. | [Nguồn](https://arxiv.org/abs/2112.10752) |
| 3 | HunyuanVideo báo cáo mô hình hơn 13 tỷ tham số; đây là đặc tả và đánh giá do nhóm tác giả công bố. | [Nguồn](https://huggingface.co/papers/2412.03603) |
| 4 | JVID kết hợp mô hình diffusion ảnh và video trong quá trình reverse diffusion để cân bằng chất lượng hình ảnh với nhất quán thời gian. | [Nguồn](https://huggingface.co/papers/2409.14149) |
| 5 | LDR tách suy luận latent dynamics và giải mã video; kết luận về ngoại suy được giới hạn trong các hệ vật lý và phép thử của bài. | [Nguồn](https://arxiv.org/html/2608.09926v1) |

### Efficient Inference and Small Language Models: A Comprehensive Survey

| Mẫu | Nội dung đã đối chiếu | Nguồn bằng chứng |
|---:|---|---|
| 1 | DistilBERT: nhỏ hơn 40%, nhanh hơn 60%, giữ 97% khả năng hiểu ngôn ngữ so với BERT trong đánh giá của bài. | [Nguồn](https://arxiv.org/abs/1910.01108) |
| 2 | MobileBERT: nhỏ hơn 4.3 lần và nhanh hơn 5.5 lần so với BERT_BASE trong đánh giá được báo cáo, không phải cam kết cho mọi thiết bị. | [Nguồn](https://aclanthology.org/2020.acl-main.195/) |
| 3 | FlashAttention là exact attention có xét IO, dùng tiling để giảm trao đổi dữ liệu HBM; không suy ra một mức tăng tốc serving chung. | [Nguồn](https://arxiv.org/abs/2205.14135) |
| 4 | QMC: abstract báo cáo giảm transfer 7.6 lần, năng lượng 11.7 lần và latency 12.5 lần so với FP16 trong cấu hình được đánh giá. | [Nguồn](https://arxiv.org/abs/2601.14549) |
| 5 | RAPID cải thiện thời gian huấn luyện RL bằng batched inference và off-policy updates; đây không phải phép đo tăng tốc inference serving. | [Nguồn](https://huggingface.co/papers/2510.03515) |

Đối chiếu thêm: PDF ICLR 2026 về Latent Particle World Models và PDF CVPR 2026 Chain of World đã được đọc; tiêu đề ICLR bị trích xuất thành “MODELS:” được agent sửa lại. Số liệu framework dùng “points” theo abstract primary v2. Lượt sửa efficient inference dùng primary QMC arXiv khi endpoint ACM PDF bị hạn chế truy cập. Tùy chọn LAB_DIRECT_PRIMARY_FETCH=1 lấy HTML công khai từ arXiv/ACL/Hugging Face ở host, có retry và che khóa; URL khác và PDF vẫn qua Exa. Sáu trang primary đã được gọi thật thành công trên Colab.

## Phạm vi ổn định và giới hạn

`recursion_limit=1000` cho lead/lượt sửa, đi kèm hạn mức model/tool; tăng số bước đồ thị không bỏ giới hạn chi phí. Researcher, citation-checker và fallback đều có hạn mức riêng. Lượt sửa tối đa hai lần sau validation; review không thay đổi nội dung được thử lại có giới hạn rồi giữ bản cũ. Lỗi đọc/ghi tệp sandbox có thông báo đã che khóa để agent tự sửa đường dẫn.

Notebook lấy cấu hình từ Colab Secrets và không hỏi nhập khóa trong lúc chạy. Drive được lưu sau mỗi chủ đề và chỉ bản đủ, hợp lệ mới được bỏ qua. Chủ đề dở dang được chạy lại. Những lần chạy mới vẫn phụ thuộc dịch vụ nguồn, hạn mức model/Daytona và kết nối 9router/tunnel; các lỗi này trả về trạng thái thất bại thay vì xuất báo cáo giả.
