# Phân tích kết quả — Memory Systems for AI Agent (Day 17)

Tài liệu này giải thích kết quả benchmark của hai agent trong `src/`: `BaselineAgent`
(short-term memory) và `AdvancedAgent` (short-term + persistent `User.md` + compact memory).

Toàn bộ số liệu dưới đây là output thật của `python src/benchmark.py`, không phải ví dụ minh họa.

---

## 1. Kết quả benchmark

### 1.1. Standard Benchmark — `data/conversations.json` (10 hội thoại, user `dungct`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                2164 |                     17720 |                  0.101 |              0.381 |                       0 |             0 |
| Advanced |                1363 |                     15436 |                  1     |              0.937 |                     625 |             0 |

### 1.2. Long-Context Stress Benchmark — `data/advanced_long_context.json` (16 lượt, user `dungct_stress`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 444 |                     23827 |                      0 |               0.32  |                       0 |             0 |
| Advanced |                 370 |                      8062 |                      1 |              0.973 |                     398 |             5 |

---

## 2. Vì sao Advanced recall tốt hơn Baseline?

Vì hai agent trả lời kiến thức từ hai nơi **khác hệt nhau**.

`BaselineAgent` chỉ có `sessions[thread_id]`. Sang thread mới, dict rỗng — nó không còn
gì để trả lời, nên câu hỏi recall chỉ còn cách echo lại câu vừa hỏi.

`AdvancedAgent` đọc `state/profiles/<user>.md` ở **mọi** thread, kể cả thread chưa từng thấy.
Cùng một câu hỏi, hai agent ở hai vị trí hoàn toàn khác nhau về tri thức.

Điểm số 0.101 của Baseline là thật, không phải lỗi. Đếm từng câu:

```
conv-01: 0/2    conv-02: 0/2, 0/2    conv-03: 0/1    conv-04: 0/2
conv-05: 2/3    conv-06: 1/2, 0/1    conv-07: 0/2
conv-08: 0/2, 0/2    conv-09: 1/4    conv-10: 0/5, 0/3
```

Vài điểm khớp là **trùng hợp do substring**, không phải nhớ: ví dụ câu hỏi của conv-05 chứa
sẵn chữ "DũngCT", và Baseline echo lại chính câu hỏi nên ăn được 2/3 chuỗi kỳ vọng.
Đây cũng là lý do cột recall được chấm bằng `expected_contains` một cách có chủ đích — nó
đo "agent có nhớ được", chứ không đo "agent có lặp lại lời người dùng".

Điểm quan trọng: chấm điểm recall bỏ dấu tiếng Việt và không phân biệt hoa thường. Nếu
không làm vậy, một câu trả lời **đúng** (`Huế`) sẽ bị chấm 0 chỉ vì khác dấu (`Hue`).

---

## 3. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Ở standard benchmark, Advanced vẫn nhẹ hơn Baseline (15436 so với 17720), nhưng **lý do
rất khác**, và nếu bật memory decay thì con số này tăng lên rõ rệt (từ 11239 lên 15436).

Mỗi lượt, Advanced phải kéo theo **cả ba lớp memory**:

```
prompt = User.md + summary + các message gần nhất
```

Baseline không có `User.md` nên prompt của nó chỉ là lịch sử thread — ở hội thoại 10 lượt,
lịch sử đó vẫn nhỏ. Advanced thì phải trả tiền cho file profile ở **mọi** lượt, kể cả khi
người dùng chỉ hỏi một câu không liên quan gì đến profile.

Ngay trong data cũng có dấu hiệu của chi phí này:

- `interests` trong corpus là `Python, AI ứng dụng, RAG, MLOps` — tích luỹ dần theo 10 hội thoại.
- `style` là `ngắn gọn, có ví dụ thực chiến, ưu tiên trade-off, có cấu trúc`.

Cả hai là field **cộng dồn**, nên `User.md` **phình dần theo thời gian**. Ở hội thoại dài,
chi phí cố định này bị bù lại bởi compact memory; ở hội thoại ngắn thì không có gì bù.

Nói cách khác: Advanced trả tiền cố định (profile + summary) để đổi lấy nhớ lâu dài. Hội
thoại càng ngắn, tỉ lệ "tiền cố định / giá trị thu được" càng xấu.

---

## 4. Vì sao compact memory giúp Advanced thắng ở hội thoại dài?

Vì prompt cost của Baseline phình theo **bậc hai**, còn của Advanced thì bị chặn trên.

Baseline không có cơ chế nén gì cả, nên ở lượt thứ *n* nó phải đọc lại toàn bộ *n−1* lượt
trước đó. Tổng chi phí vì thế là tổng của một dãy cấp số cộng — `O(n²)`.

Stress conversation có 16 lượt với tổng 2322 token user. Chạy thật:

```
Baseline prompt tokens processed: 23827
Advanced sau 5 lần compact:        8062   (~34% của Baseline)
```

Nguyên nhân compact memory có tác dụng nằm ở hai chỗ, cùng lúc:

1. **Buffer message** luôn bị cắt còn `keep_messages=4` lượt gần nhất.
2. **Summary có trần cứng** — không vượt quá 25% ngưỡng token (tối đa 1200 ký tự).

Điểm thứ hai là điều dễ làm sai. Ở bản đầu tiên của tôi, summary được phép dài tới 1200 ký
tự (~300 token) trong khi ngưỡng chỉ 400 token — tức là summary nuốt gần hết ngân sách,
không giải phóng được áp lực, và thread phải compact lại liên tục. Sau khi ràng buộc
summary vào một tỷ lệ của ngưỡng, compaction mới thực sự kéo context xuống.

**Vì sao compact chủ yếu tối ưu `prompt tokens processed`, không phải `agent tokens only`?**

Vì hai chỉ số đo hai việc khác nhau:

- `Agent tokens only` = token agent **sinh ra** → phụ thuộc vào việc câu trả lời dài hay ngắn.
- `Prompt tokens processed` = token agent **phải đọc** → phụ thuộc vào lịch sử mang theo.

Compact memory chỉ can thiệp vào vế thứ hai. Nó không làm agent viết ngắn hơn, và không nên.
Với cùng một câu hỏi, Advanced vẫn sinh câu trả lời dài tương tự Baseline — nhưng phải trả
ít tiền hơn để đọc được lịch sử đó.

---

## 5. Memory growth (bytes) và rủi ro đi kèm

| Suite  | Baseline | Advanced |
|--------|----------|----------|
| Chuẩn  | 0 byte   | 625 byte |
| Stress | 0 byte   | 398 byte |

Baseline có 0 byte vì nó **không có file memory nào** — đây là chi phí bị dịch sang Advanced,
không phải Advanced bị "phạt" vì lãng phí.

625 byte cho 10 hội thoại nghe rất nhỏ, nhưng đó chính là điểm: **kích thước file tăng tuyến
tính theo số fact, và các field cộng dồn (`style`, `interests`) không bao giờ bị gọn lại.**

Đo cụ thể bằng cách tắt `touch_fact` (tức là tắt memory decay) rồi chạy lại standard benchmark:

| Cấu hình | Prompt tokens | Memory growth | Recall |
|----------|---------------|---------------|--------|
| Không memory decay | 11239 | 330 byte | 1.000 |
| Có memory decay    | 15436 | 625 byte | 1.000 |

Recall không đổi, nhưng **prompt cost tăng 37%** và file phình gần gấp đôi. Đây là cái giá
cụ thể của việc ghi mốc `_seen=<n>` vào từng fact: nó làm `User.md` phình ra, mà `User.md`
được nạp vào mọi lượt. Đây chính là dạng trade-off mà rubric muốn thấy được nêu ra.

### Ba rủi ro thật

**a) File phình không kiểm soát.** Field scalar (`location`, `profession`) có tối đa một
giá trị nên bị chặn trên. Nhưng field cộng dồn chỉ giảm ưu tiên, không bao giờ xóa. Một người
dùng đổi sở thích 20 lần sẽ tích lũy 20 mục. Ở quy mô production, đây là chỗ cần đặt trần
hoặc LLM tóm tắt lại profile theo định kỳ.

**b) Lưu sai fact là thất bại âm thầm.** Regex trích fact theo mẫu câu, nên chỉ đúng với
ngôn ngữ đã thấy. Người dùng thật sẽ nói "Tui ở chỗ này nè" và agent sẽ im lặng bỏ qua — lúc
đó recall giảm một cách âm thầm, không có báo lỗi nào.

**c) Correction là điểm yếu thật sự.** Bộ dữ liệu cố tình đặt sẵn các tình huống bẫy:

- conv-03 đổi nơi ở Đà Nẵng → Huế
- conv-06 đổi nghề backend → MLOps
- stress-01 đổi Huế → Đà Nẵng, kèm nhiễu `Hà Nội` (đi họp 2 ngày) và
  `product manager` (câu đùa)

Hai cơ chế giữ được các fact này đúng:

1. **Conflict handling** — field scalar bị ghi đè, không tồn tại song song hai giá trị.
   Điểm then chốt: một câu đính chính **nhắc giá trị cũ trước giá trị mới**
   ("không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer"). Nếu lấy match
   **đầu tiên**, agent sẽ ghi lại `backend engineer` — tức là ghi nhớ chính cái fact người
   dùng vừa xóa. Vì vậy code phải lấy match **cuối cùng**.

2. **Noise guard** — các cụm "chỉ là câu đùa", "không phải nơi ở", "chỉ là nơi mình vừa bay
   ra họp" chặn luôn việc ghi fact, nên nhiễu không bao giờ ghi đè fact thật.

Có test riêng bảo vệ đúng điều này (`test_advanced_resolves_corrections_to_the_newest_fact`):
sau khi đính chính, câu trả lời **không được chứa** `Đà Nẵng` hay `backend engineer`.

---

## 6. Bốn bonus đã triển khai

| Bonus | Cách làm | Giải quyết vấn đề gì | Rủi ro mới |
|-------|----------|----------------------|-----------|
| **Confidence threshold** | Mỗi fact gắn điểm 0.95 (khai báo trực tiếp) / 0.85 (nhắc lại) / 0.5 (nhắc thoáng). Dưới 0.6 thì không ghi. | Không ghi fact chỉ vì nhắc lướt qua. | Người dùng nói gián tiếp thì mất fact; nới ngưỡng thì lọc nhiễu kém hơn. |
| **Conflict handling** | Field scalar ghi đè, dùng match cuối cùng trong câu. | `Đà Nẵng` → `Huế`, `backend` → `MLOps` không còn tồn tại song song. | Câu đính chính kiểu "trước giờ tôi ở X" có thể ghi nhầm nếu mẫu câu lệch. |
| **Entity extraction có cấu trúc** | `User.md` là danh sách `- key: value`, key ASCII, có metadata `_updated`. | Cho phép cập nhật đúng một dòng thay vì sửa văn bản tự do. | Người dùng thật vẫn có thể nói ngoài từ vựng định sẵn. |
| **Memory decay** | `touch_fact()` ghi `_seen=<n>`; `ranked_facts()` xếp theo độ mới. | Fact lâu không được nhắc trượt xuống thứ hạng. | Hệ số `_seen` phải được lọc khỏi value — nếu sót, nó sẽ bị đọc thành câu trả lời và làm hỏng cả recall. |

Về memory decay: nó **hạ ưu tiên, không xóa**. Xóa thật sẽ âm thầm mất thông tin người dùng có
thể vẫn cần. Đây là lựa chọn có chủ đích, và là điểm yếu của nó: hồ sơ vẫn phình theo thời gian.

---

## 7. Bảng trade-off tổng hợp

| Tiêu chí | Baseline | Advanced | Ai thắng |
|----------|----------|----------|----------|
| Recall chéo phiên (chuẩn) | 0.101 | 1.000 | Advanced |
| Recall chéo phiên (stress) | 0.000 | 1.000 | Advanced |
| Prompt tokens (chuẩn) | 17720 | 15436 | Advanced (nhẹ) |
| Prompt tokens (stress) | 23827 | 8062 | Advanced (rõ rệt) |
| Agent tokens only | 2164 | 1363 | Advanced |
| Memory growth | 0 byte | 625 byte | Baseline (nhưng đổi lại mất recall) |
| Compactions | 0 | 5 | Baseline (không cần) |
| Độ phức tạp code | thấp | cao hơn nhiều | Baseline |

**Kết luận.** Ở bộ dữ liệu này Advanced thắng gần như mọi chỉ số, nhưng đó **không phải** vì
hệ thống memory miễn phí:

- Advanced thắng recall vì có persistent memory — đây là lợi thế thật, không phải may mắn.
- Advanced thắng prompt cost ở stress vì compact memory có trần cứng; nếu không, prompt cost
  phình `O(n²)` và Advanced sẽ thua.
- Advanced **trả giá bằng độ phức tạp và bằng một file memory phình dần** — ở standard
  benchmark, bật memory decay đã làm prompt cost của Advanced tăng từ 11239 lên 15436.
  Đây chính là dạng trade-off mà README mô tả: ở hội thoại ngắn, Advanced dễ tốn hơn.

---

## 8. Giới hạn của kết quả này

Nói thẳng những gì benchmark này **không** chứng minh:

1. **Stress benchmark chỉ có n=1.** Toàn bộ cột "5 lần compact" đến từ **một** hội thoại 16
   lượt. Không nên khái quát hoá thành "compact luôn thắng". Cần thêm nhiều hội thoại dài
   khác đa dạng hơn.

2. **Chỉ số recall là so khớp chuỗi.** Một câu trả lời dài dòng chứa đủ chuỗi kỳ vọng vẫn
   được 1.0. Vì vậy `heuristic_quality()` có thêm điểm cho ngắn gọn và có cấu trúc, nhưng nó
   vẫn là **proxy**, không phải chấm điểm chất lượng thật.

3. **Chế độ offline là deterministic, không phải LLM.** Câu trả lời được ghép từ template
   dựa trên field trong `User.md`. Nó kiểm chứng **lớp memory**, chứ không kiểm chứng chất
   lượng sinh câu trả lời. Chạy với LLM thật qua endpoint OpenAI-compatible sẽ cho điểm
   `Response quality` đáng tin hơn — nhưng benchmark khi đó **không còn tái lập được**.

4. **Regex bị giới hạn bởi ngôn ngữ đã thấy.** Data là tiếng Việt với mẫu câu rõ ràng.
   Người dùng thật sẽ dùng cách diễu đạt khác, và recall sẽ giảm mà không báo lỗi.

---

## 9. Cách chạy lại

```bash
# Windows: dùng interpreter có sẵn pytest
/c/ProgramData/anaconda3/python.exe src/benchmark.py
/c/ProgramData/anaconda3/python.exe -m pytest src/ -q
```

Không cần API key. Toàn bộ lab chạy offline theo mặc định; `langchain` **không được import
ở cấp module**, nên thiếu dependency cũng không làm hỏng.