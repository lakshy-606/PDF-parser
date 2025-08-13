import os
import sys
import re
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import faiss
import pickle

load_dotenv()

def is_table_line(line):
    """Check if line is part of markdown table"""
    line = line.strip()
    return (line.count('|') >= 2 and
            not line.startswith(('#', '```', '>', '-', '*', '+')))

def extract_tables(text):
    """Find all markdown tables"""
    lines = text.split('\n')
    tables = []
    start = None
    table_lines = []

    for i, line in enumerate(lines):
        if is_table_line(line):
            if start is None:
                start = i
            table_lines.append(line)
        else:
            if start is not None and len(table_lines) >= 2:
                tables.append((start, i-1, '\n'.join(table_lines)))
            start = None
            table_lines = []

    if start is not None and len(table_lines) >= 2:
        tables.append((start, len(lines)-1, '\n'.join(table_lines)))

    return tables

def is_heading(line):
    """Check if line is a markdown heading"""
    return line.strip().startswith('#')

def should_keep_with_next(line, next_lines):
    """Determine if current line should be kept with following content"""
    if is_heading(line):
        return True

    # Keep short lines (likely titles or important context) with following content
    if len(line.strip()) < 100 and next_lines:
        # Check if next non-empty line is content (not another heading)
        for next_line in next_lines[:3]:  # Look ahead up to 3 lines
            if next_line.strip():
                if not is_heading(next_line):
                    return True
                break
    return False

def get_context_overlap(text, char_limit=50):
    """Get the last char_limit characters from text as context"""
    if not text or len(text) <= char_limit:
        return text

    # Try to break at word boundary
    truncated = text[-char_limit:]
    space_index = truncated.find(' ')
    if space_index > 0:
        return truncated[space_index:].strip()
    return truncated.strip()

def chunk_markdown(content, file_name):
    """Improved chunking that preserves context and keeps headings with content"""
    lines = content.split('\n')
    tables = extract_tables(content)

    # Mark table lines
    table_line_nums = set()
    for start, end, _ in tables:
        table_line_nums.update(range(start, end + 1))

    chunks = []
    current_chunk = []
    previous_text_content = ""  # Track previous content for table context
    i = 0

    while i < len(lines):
        line = lines[i]

        # If line is part of table, handle table with context
        if i in table_line_nums:
            # Save current chunk if exists
            if current_chunk:
                text = '\n'.join(current_chunk).strip()
                if len(text) > 10:
                    chunks.append({'text': text, 'type': 'text', 'file_name': file_name})
                    previous_text_content = text
                current_chunk = []

            # Find and add the complete table with context
            for start, end, table_content in tables:
                if start <= i <= end:
                    # Add context overlap before table
                    context = get_context_overlap(previous_text_content, 200)
                    if context:
                        table_with_context = f"{context}\n\n{table_content}"
                    else:
                        table_with_context = table_content

                    chunks.append({
                        'text': table_with_context, 
                        'type': 'table', 
                        'file_name': file_name
                    })
                    i = end + 1
                    break
            continue

        # Handle regular content with improved chunking logic
        if line.strip() == '':
            # Only chunk on empty line if we're not keeping heading with content
            if current_chunk:
                # Check if last line should be kept with next content
                last_line = current_chunk[-1] if current_chunk else ""
                upcoming_lines = lines[i+1:i+4] if i+1 < len(lines) else []

                if not should_keep_with_next(last_line, upcoming_lines):
                    text = '\n'.join(current_chunk).strip()
                    if len(text) > 10:
                        chunks.append({'text': text, 'type': 'text', 'file_name': file_name})
                        previous_text_content = text
                    current_chunk = []
                else:
                    # Keep the content together, don't chunk here
                    current_chunk.append(line)
            else:
                # Empty chunk, just skip empty line
                pass
        else:
            current_chunk.append(line)

        i += 1

    # Add final chunk
    if current_chunk:
        text = '\n'.join(current_chunk).strip()
        if len(text) > 10:
            chunks.append({'text': text, 'type': 'text', 'file_name': file_name})

    return chunks

def create_embeddings(chunks):
    """Create and store embeddings"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts = [chunk['text'] for chunk in chunks]
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = np.array([v.embedding for v in response.data], dtype=np.float32)

    index = faiss.IndexFlatIP(1536)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, "faiss_index.bin")

    with open("metadata.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Created {len(chunks)} chunks")
    table_count = sum(1 for c in chunks if c['type'] == 'table')
    print(f" - {table_count} tables, {len(chunks)-table_count} text chunks")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py file.md [name]")
        return

    file_path = sys.argv[1]
    file_name = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(file_path)

    if not file_path.endswith('.md'):
        print("Error: Only .md files supported")
        return

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = chunk_markdown(content, file_name)
    create_embeddings(chunks)

    print("Done! Created: faiss_index.bin, metadata.pkl")

if __name__ == "__main__":
    main()
