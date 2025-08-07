import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title, ListItem, Table
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set up paths for Windows (adjust these paths based on your installation)
os.environ["PATH"] += os.pathsep + r"C:\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

class PDFParser:
    """
    PDF Parser that extracts all content using Unstructured and saves to text file
    """

    def __init__(self):
        print("[INFO] PDF Parser initialized")

    def process_pdf(self, pdf_path, output_text_file="extracted_content.txt"):
        """
        Process PDF using Unstructured and save all content to text file

        Args:
            pdf_path: Path to PDF file
            output_text_file: Output text file path
        """
        print(f"[INFO] Processing: {pdf_path}")

        try:
            # Process PDF with high-resolution strategy for best layout detection
            elements = partition_pdf(
                filename=pdf_path,
                strategy="hi_res",  # Best for complex layouts and tables
                infer_table_structure=True,  # Detect and preserve table structure
                pdf_infer_table_structure=True,  # PDF-specific table inference
                ocr_languages="eng",  # OCR language
                extract_images_in_pdf=False,  # Skip images for text-focused processing
            )

        except Exception as e:
            print(f"[ERROR] Failed to process PDF: {e}")
            return False

        print(f"[INFO] Extracted {len(elements)} elements from PDF")

        # Extract and format content
        content_lines = []

        for el in elements:
            # Only process text-based elements
            if isinstance(el, (NarrativeText, Title, ListItem, Table)):
                text = el.text
                if text and text.strip():  # Skip empty elements

                    # Get page number if available
                    page_num = getattr(el.metadata, 'page_number', None) if hasattr(el, 'metadata') else None
                    element_type = type(el).__name__

                    # Format content with metadata
                    content_lines.append(f"[PAGE: {page_num}] [TYPE: {element_type}]")
                    content_lines.append(text.strip())
                    content_lines.append("")  # Empty line for separation

        # Save to text file
        try:
            with open(output_text_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(content_lines))

            print(f"[SUCCESS] Content saved to: {output_text_file}")
            print(f"[INFO] Total elements processed: {len([l for l in content_lines if l.startswith('[PAGE:')])}")
            print(f"[INFO] File size: {os.path.getsize(output_text_file)} bytes")

            return True

        except Exception as e:
            print(f"[ERROR] Failed to save content: {e}")
            return False

def main():
    """Main function to parse PDF"""
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <pdf_path> [output_file]")
        print("Example: python pdf_parser.py document.pdf extracted_content.txt")
        return

    pdf_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "extracted_content.txt"

    if not os.path.exists(pdf_path):
        print(f"[ERROR] PDF file not found: {pdf_path}")
        return

    parser = PDFParser()
    success = parser.process_pdf(pdf_path, output_file)

    if success:
        print(f"\n[NEXT STEP] Run the chunker script:")
        print(f"python text_chunker_embedder.py {output_file}")
    else:
        print("[ERROR] PDF processing failed.")

if __name__ == "__main__":
    main()
