# PdfToEpub — V4 LOCAL_TURBO

Pipeline chuyển PDF scan hai trang → OCR tiếng Việt → TXT/EPUB, ưu tiên **tốc độ local + khả năng audit**, sau đó có thể chạy **DeepSeek validator riêng** mà **không OCR lại**.

## Hai mode độc lập

### 1. LOCAL_TURBO

- Render từng PDF spread và tách trái/phải theo geometry.
- 5 whole-side OCR evidence passes cho mọi side.
- **Whole-side health detector** bắt trường hợp OCR sập toàn trang (symbol soup, fragmented words, cross-pass instability, confidence/lexicon collapse).
- Chỉ side bị đánh dấu catastrophe mới chạy thêm 4 fallback passes với preprocessing/PSM khác.
- Nếu fallback vẫn thất bại, side được ghi vào `whole_side_health.json` + `local_review.json`; line của side đó **không được gửi sang DeepSeek để vá lẻ tẻ**.
- Với side khỏe, chỉ các line đáng nghi mới reOCR từ source pixels.
- Local correction chỉ nhận thay đổi có visual consensus; không dùng AI.
- Cleanup header/page number/ornament theo rule rõ ràng.
- Xuất TXT, EPUB và audit JSON.

### 2. DEEP_ONLY

- Đọc lại `*_V4_LOCAL_TURBO.txt` + `local_refine_audit.json` đã có.
- **Không mở PDF, không gọi Tesseract.**
- Queue các line còn đáng nghi, ngoại trừ line thuộc whole-side catastrophe chưa cứu được.
- Micro-batch mặc định **6 items/call**, **4 OpenCode/DeepSeek calls song song**.
- Model mặc định: `opencode/deepseek-v4-flash-free`.
- DeepSeek chỉ **đề xuất** correction; `deep/gate.py` quyết định apply.

### Evidence-aware gate

Gate không còn dùng `confidence >= 0.97` như công tắc cứng:

- 2+ OCR alternatives cùng ủng hộ NEW → threshold thấp hơn mạnh nhưng không dưới 0.90.
- 1 OCR alternative ủng hộ NEW → mặc định có thể apply từ ~0.95.
- Sửa chỉ khác dấu/shape được ưu tiên hơn vì giữ nguyên glyph sequence.
- Không có visual support và không giữ shape → chặn, kể cả AI confidence rất cao.
- Replacement dùng **token boundary**, không dùng `str.count`, nên `kế` không bị nhầm với substring trong `kết`.

### Word segmentation

Deep có operation riêng `kind="segment"` cho lỗi OCR dính từ:

```text
Vidu  →  Ví dụ
```

Segmentation chỉ được apply khi:

- OCR alternative có đúng phrase đã tách; **hoặc**
- bỏ space + dấu tiếng Việt thì OLD và NEW có cùng glyph shape.

Không cho phép dùng segmentation để chèn/rewrite từ không liên quan.

## Cấu trúc

```text
src/pdf_to_epub/
├─ cli.py
├─ config.py
├─ models.py
├─ pdf_layout.py
├─ pipeline.py
├─ cleanup.py
├─ epub.py
├─ jsonio.py
├─ logging_utils.py
├─ ocr/
│  ├─ preprocess.py
│  ├─ tesseract.py          # fast + catastrophe fallback pass definitions
│  ├─ batch.py              # image-list scheduler
│  ├─ scoring.py            # shared OCR shape/quality helpers
│  ├─ health.py             # whole-side catastrophe policy
│  ├─ lexicon.py
│  └─ local_refine.py
└─ deep/
   ├─ queue.py
   ├─ prompt.py
   ├─ client.py
   ├─ gate.py               # evidence-aware + segmentation safety policy
   └─ polish.py
```

Nguyên tắc phân nhiệm: **module gọi OCR/model không quyết định policy**, và **policy module không tự gọi service**. `health.py` quyết định *khi nào* retry whole side; `batch.py` chỉ thực thi retry. `gate.py` quyết định correction; `client.py` chỉ gọi OpenCode.

## Cài trên Windows

Yêu cầu Python 3.12+, Tesseract có `vie` + `eng`; OpenCode chỉ cần cho `--deep-only`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

Nếu Tesseract không nằm trong PATH:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Chạy local OCR

```powershell
pdf-to-epub .\pdf.pdf --start 61 --end 100
```

Output quan trọng:

```text
pdf_v4_local_turbo_61_100/
├─ pdf_PDF_61_100_V4_LOCAL_TURBO.txt
├─ pdf_PDF_61_100_V4_LOCAL_TURBO.epub
├─ whole_side_health.json
├─ local_refine_audit.json
├─ local_review.json
├─ cleanup_audit.json
└─ SUMMARY_V4_LOCAL_TURBO.json
```

## Chạy riêng DeepSeek, không OCR lại

```powershell
pdf-to-epub .\pdf.pdf --start 61 --end 100 --deep-only `
  --ai-batch-size 6 `
  --ai-workers 4 `
  --model "opencode/deepseek-v4-flash-free"
```

LOCAL TXT/EPUB vẫn nguyên vẹn; Deep tạo file `_DEEP.txt/.epub` riêng.

## Runner 1-click

```text
scripts\run_local_turbo.bat
scripts\run_deep_only.bat
```

## Test

```powershell
python -m pip install pytest
pytest -q
```

Tests khóa các boundary dễ phá text nhất: evidence-aware Deep gate, token uniqueness, safe word segmentation, whole-side catastrophe detection/fallback selection và EPUB serialization.

## Không commit dữ liệu sách

`.gitignore` loại trừ PDF, EPUB, TXT OCR và toàn bộ runtime audit/output chứa nội dung sách. Repo chỉ lưu engine/code/test.
