# PdfToEpub — V4 LOCAL_TURBO baseline

Pipeline chuyển PDF scan hai trang → OCR tiếng Việt → TXT/EPUB, ưu tiên **tốc độ local + khả năng audit**, sau đó có thể chạy **DeepSeek validator riêng** mà **không OCR lại**.

Đây là mốc code trước khi chỉnh tiếp safety gate/whole-side failure handling. Cấu trúc được tách module để những lần sửa sau không làm dây chuyền ảnh hưởng toàn pipeline.

## Hai mode độc lập

### 1. LOCAL_TURBO

- Render từng PDF spread và tách trái/phải theo geometry.
- 5 whole-side OCR evidence passes.
- Chọn pass tốt nhất bằng confidence/length/garbage score.
- Chỉ các line đáng nghi mới reOCR từ source pixels.
- ReOCR line được gom thành image-list batch khoảng 28 line/process để giảm process-launch overhead.
- Local correction chỉ nhận thay đổi có visual consensus; không dùng AI.
- Cleanup header/page number/ornament theo rule rõ ràng.
- Xuất TXT, EPUB và audit JSON.

Baseline test trước khi đưa repo: 40 PDF pages (61..100), 80 logical sides, 22 workers, `batch_sides=2`.

### 2. DEEP_ONLY

- Đọc lại `*_V4_LOCAL_TURBO.txt` + `local_refine_audit.json` đã có.
- **Không mở PDF, không gọi Tesseract.**
- Queue các line còn đáng nghi.
- Micro-batch mặc định **6 items/call**.
- Chạy **4 OpenCode/DeepSeek calls song song**.
- Model mặc định: `opencode/deepseek-v4-flash-free`.
- Model chỉ đề xuất sửa **một OCR token**; safety gate local quyết định có apply hay không.
- Không ghi đè LOCAL output; tạo riêng `*_DEEP.txt/.epub` và audit.

## Cấu trúc

```text
src/pdf_to_epub/
├─ cli.py                 # CLI, chỉ parse option và chọn mode
├─ config.py              # Toàn bộ default/config runtime
├─ models.py              # Contract dữ liệu giữa các stage
├─ pdf_layout.py          # Render + split spread L/R
├─ pipeline.py            # Orchestrator LOCAL_TURBO
├─ cleanup.py             # Header/page-number/ornament cleanup
├─ epub.py                # Write + validate EPUB3
├─ jsonio.py              # JSON/TXT IO nhỏ, dùng chung
├─ logging_utils.py       # Run logger
├─ ocr/
│  ├─ preprocess.py       # Gray/sharp/resize/threshold
│  ├─ tesseract.py        # Tesseract config + pass definitions
│  ├─ batch.py            # Image-list batching/scheduler
│  ├─ scoring.py          # OCR quality/garbage/diacritic helpers
│  ├─ lexicon.py          # Vietnamese DAWG lexicon, best effort
│  └─ local_refine.py     # Local evidence-driven correction policy
└─ deep/
   ├─ queue.py            # Build unresolved-line queue
   ├─ prompt.py           # Strict OCR-only prompt
   ├─ client.py           # OpenCode CLI adapter + JSON parser
   ├─ gate.py             # Safety policy: nơi sửa gate về sau
   └─ polish.py           # Parallel Deep-only orchestration + patch
```

Nguyên tắc phân nhiệm: **module gọi dịch vụ không quyết định policy**, và **module quyết định policy không tự chạy OCR/model**. Nhờ vậy việc tune một tầng có phạm vi thay đổi nhỏ và test được độc lập.

## Cài trên Windows

Yêu cầu:

- Python 3.12+
- Tesseract OCR có `vie` và `eng`
- OpenCode CLI chỉ cần khi chạy `--deep-only`

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

Đặt PDF bất kỳ, ví dụ `pdf.pdf`:

```powershell
pdf-to-epub .\pdf.pdf --start 61 --end 100
```

Hoặc:

```powershell
python -m pdf_to_epub .\pdf.pdf --start 61 --end 100
```

Output mặc định:

```text
pdf_v4_local_turbo_61_100/
├─ pdf_PDF_61_100_V4_LOCAL_TURBO.txt
├─ pdf_PDF_61_100_V4_LOCAL_TURBO.epub
├─ local_refine_audit.json
├─ local_review.json
├─ cleanup_audit.json
├─ SUMMARY_V4_LOCAL_TURBO.json
└─ run_v4_local_turbo.log
```

## Chạy riêng DeepSeek, không OCR lại

Sau khi LOCAL_TURBO đã chạy xong:

```powershell
pdf-to-epub .\pdf.pdf --start 61 --end 100 --deep-only `
  --ai-batch-size 6 `
  --ai-workers 4 `
  --model "opencode/deepseek-v4-flash-free"
```

Output thêm:

```text
pdf_PDF_61_100_V4_LOCAL_TURBO_DEEP.txt
pdf_PDF_61_100_V4_LOCAL_TURBO_DEEP.epub
deep_ai_queue.json
deep_ai_audit.json
SUMMARY_DEEP_ONLY.json
run_deep_only.log
```

LOCAL TXT/EPUB vẫn nguyên vẹn để luôn có thể so sánh trước/sau AI.

## Runner 1-click

Nếu file tên `pdf.pdf` nằm ở root repo:

```text
scripts\run_local_turbo.bat
scripts\run_deep_only.bat
```

Muốn đổi range, sửa `START_PAGE` / `END_PAGE` ngay đầu BAT hoặc truyền CLI trực tiếp.

## Audit và nguyên tắc an toàn

DeepSeek **không được quyền rewrite câu**. Mỗi operation phải:

1. có `old` và `new` là một token, không có whitespace;
2. `old` phải xuất hiện đúng một lần theo exact-substring rule hiện tại;
3. confidence mặc định phải `>= 0.97`;
4. `new` phải có OCR alternative support hoặc chỉ khác dấu/case;
5. tối đa 3 operation/item.

Mọi operation, kể cả bị chặn, đều ghi `gate` + `applied` vào `deep_ai_audit.json`.

## Không commit dữ liệu sách

`.gitignore` loại trừ PDF, EPUB, TXT OCR, prompt runtime và audit/output runtime. Repo chỉ lưu engine/code/test; tránh đưa nội dung sách hoặc file test có bản quyền lên GitHub public.

## Test

```powershell
python -m pip install pytest
pytest -q
```

Các test hiện tập trung vào những boundary dễ gây phá text nhất: Deep safety gate và EPUB serialization/validation.
