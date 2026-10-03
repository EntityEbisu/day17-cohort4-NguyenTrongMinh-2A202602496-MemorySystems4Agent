# Day 17 — Memory Systems for AI Agent (bài làm hoàn chỉnh)

Thư mục `src/` này là bài làm đã hoàn thành cho lab Day 17. Toàn bộ chạy **offline theo mặc
định**: không cần API key, không cần LangChain, kết quả tái lập được.

## Cấu trúc

| File | Vai trò |
|---|---|
| `model_provider.py` | Chuẩn hoá tên provider + khởi tạo chat model (import lazy) |
| `config.py` | Đường dẫn, ngưỡng compact, provider cho model chính và judge |
| `memory_store.py` | Lớp memory: ước lượng token, `User.md`, trích fact, compact memory |
| `agent_baseline.py` | Agent A: chỉ nhớ trong cùng thread |
| `agent_advanced.py` | Agent B: short-term + `User.md` + compact memory |
| `benchmark.py` | Chạy 2 bộ benchmark và in 6 cột so sánh |
| `test_*.py` | 50 test cho hành vi memory |

## Ba lớp memory

- **Short-term** — lịch sử trong thread hiện tại, bị giới hạn bởi compact memory.
- **Persistent** — `state/profiles/<user_id>.md`, sống qua nhiều thread và cả các lần chạy.
- **Compact** — khi thread vượt ngưỡng token, lịch sử cũ được nén vào summary có trần cứng.

## Chạy

```bash
# Benchmark (in 2 bảng + nhận xét)
python src/benchmark.py

# Test
pytest src/ -v
```

Không cần biến môi trường. Nếu muốn chạy chế độ live với LLM thật, tạo file `.env` ở
thư mục gốc (đã nằm trong `.gitignore`):

```
LLM_PROVIDER=custom
LLM_MODEL=<tên model>
CUSTOM_BASE_URL=http://127.0.0.1:1234/v1
CUSTOM_API_KEY=<key>
```

Biến tùy chọn: `COMPACT_THRESHOLD_TOKENS` (mặc định 600), `COMPACT_KEEP_MESSAGES` (4),
`JUDGE_MODEL`, `LLM_TEMPERATURE`.

## Provider được hỗ trợ

`openai`, `custom` (endpoint tương thích OpenAI), `gemini`, `anthropic`, `ollama`, `openrouter`.

Lưu ý khi đọc code: **không có import `langchain` ở cấp module**. Toàn bộ import provider
nằm trong từng nhánh của `build_chat_model()` và được bọc `try/except ImportError`, nên thiếu
dependency không làm hỏng chế độ offline.

Dữ liệu benchmark nằm ở `data/` ở thư mục gốc. Phần phân tích kết quả nằm ở `ANALYSIS.md`.
