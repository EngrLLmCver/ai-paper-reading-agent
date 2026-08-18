import sys
from docx import Document

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

path = sys.argv[1]
doc = Document(path)
for para in doc.paragraphs:
    if para.text.strip():
        print(para.text)
