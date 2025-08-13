#!/usr/bin/env python3
import os, sys, time, argparse, logging
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from openai import OpenAI

load_dotenv()

os.environ["PATH"] += os.pathsep + r"C:\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_INPUT_TOKENS = 20000
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_BATCH_PAGES = 4
TOKENS_PER_CHAR = 0.25
RETRY_LIMIT = 3
RETRY_DELAY_BASE = 2.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()
client = OpenAI()

def estimate_tokens(text): return int(len(text) * TOKENS_PER_CHAR)

def extract_pages(pdf_path: str) -> Dict[int, str]:
    elements = partition_pdf(
        filename=pdf_path, strategy="auto", 
        infer_table_structure=True, pdf_infer_table_structure=True,
        ocr_languages="eng", extract_images_in_pdf=False
    )
    pages: Dict[int, List[str]] = {}
    for el in elements:
        pnum = getattr(el.metadata, "page_number", 1)
        pages.setdefault(pnum, []).append(str(el))
    return {p: "\n".join(txts) for p, txts in sorted(pages.items())}

def batch_text(batch_pages: List[Tuple[int, str]]) -> str:
    return "".join([f"\n\n--- PAGE {p} START ---\n{txt}\n--- PAGE {p} END ---\n" for p, txt in batch_pages])

def sys_prompt() -> str:
    return (
        "You are a precise PDF-to-Markdown converter. Rules:\n"
        "- Output ONLY Markdown\n"
        "- Preserve wording & punctuation\n"
        "- Rejoin broken lines/hyphens\n"
        "- Remove headers/footers/page numbers\n"
        "- Maintain reading order\n"
        "- Preserve headings, lists, tables\n"
        "- Convert tables to Markdown tables\n"
        "- Do NOT add or summarize content"
    )

def usr_prompt(text: str) -> str:
    return f"Convert the following PDF content to Markdown:\n\n{text}\n"

def gpt_call(model, system_prompt, user_prompt, max_output_tokens):
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=max_output_tokens,
    )
    return r.choices[0].message.content.strip()

def gpt_call_retry(*args, **kwargs):
    delay = RETRY_DELAY_BASE
    for attempt in range(RETRY_LIMIT):
        try:
            return gpt_call(*args, **kwargs)
        except Exception as e:
            if attempt == RETRY_LIMIT - 1: raise
            logger.warning("GPT call failed (%s). Retrying in %.1fs…", e, delay)
            time.sleep(delay); delay *= 2

def plan_batches(pages, max_input_tokens, start_size):
    base = estimate_tokens(sys_prompt())
    items, i, batches = list(pages.items()), 0, []
    while i < len(items):
        size = start_size
        while size > 0:
            candidate = items[i:i+size]
            toks = base + estimate_tokens(usr_prompt(batch_text(candidate)))
            if toks <= max_input_tokens:
                batches.append(candidate); i += size; break
            size -= 1
        if size == 0:
            batches.append(items[i:i+1]); i += 1
    return batches

def convert(pdf, out_dir, out_md, model, max_in, max_out, batch_pages):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pages = extract_pages(pdf)
    batches = plan_batches(pages, max_in, batch_pages)
    sys_msg, parts = sys_prompt(), []
    for idx, batch in enumerate(batches, 1):
        file = Path(out_dir)/f"batch_{idx:03d}.md"
        if file.exists():
            content = file.read_text(encoding="utf-8")
        else:
            content = gpt_call_retry(model, sys_msg, usr_prompt(batch_text(batch)), max_out)
            file.write_text(content, encoding="utf-8")
        parts.append(content)
    Path(out_md).write_text("\n\n".join(parts).strip(), encoding="utf-8")
    logger.info("✅ Conversion complete → %s", out_md)

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set"); sys.exit(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out_dir", default="pdf_form")
    ap.add_argument("--out_md", default="output_form.md")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_input_tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    ap.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    ap.add_argument("--batch_pages", type=int, default=DEFAULT_BATCH_PAGES)
    args = ap.parse_args()
    try:
        convert(args.pdf, args.out_dir, args.out_md, args.model,
                args.max_input_tokens, args.max_output_tokens, args.batch_pages)
    except Exception as exc:
        logger.error("❌ %s", exc); sys.exit(1)

if __name__ == "__main__":
    main()
