"""
论文文本提取工具
支持 PDF / DOCX / TXT / MD 格式
输出：纯文本文件，保留段落结构
用法：python extract_paper.py <文件路径> [--output <输出路径>]
"""
import sys
import os
import argparse
import re

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def extract_pdf(filepath: str) -> str:
    """从PDF提取文本"""
    from PyPDF2 import PdfReader

    reader = PdfReader(filepath)
    full_text = []
    total_pages = len(reader.pages)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # 清理多余空白
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            text = re.sub(r'\n{3,}', '\n\n', text)
            full_text.append(f"--- 第 {i+1}/{total_pages} 页 ---\n{text}")
        else:
            full_text.append(f"--- 第 {i+1}/{total_pages} 页 ---\n[无法提取文本，可能是扫描版PDF]")

    return "\n\n".join(full_text)


def extract_docx(filepath: str) -> str:
    """从DOCX提取文本"""
    from docx import Document

    doc = Document(filepath)
    paragraphs = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            paragraphs.append("")
            continue

        # 根据样式判断是否是标题
        style_name = para.style.name if para.style else ""
        if "Heading" in style_name or "heading" in style_name or "标题" in style_name:
            level = re.search(r'\d+', style_name)
            lvl = level.group() if level else "1"
            prefix = "#" * min(int(lvl), 6)
            paragraphs.append(f"\n{prefix} {text}\n")
        else:
            paragraphs.append(text)

    return "\n".join(paragraphs)


def extract_txt(filepath: str) -> str:
    """从纯文本/MD文件读取"""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def detect_type(filepath: str) -> str:
    """根据扩展名判断文件类型"""
    ext = os.path.splitext(filepath)[1].lower()
    type_map = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".txt": "txt",
        ".md": "txt",
        ".markdown": "txt",
    }
    return type_map.get(ext, "unknown")


def extract(filepath: str) -> str:
    """主提取函数"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    file_type = detect_type(filepath)

    if file_type == "pdf":
        text = extract_pdf(filepath)
    elif file_type == "docx":
        text = extract_docx(filepath)
    elif file_type == "txt":
        text = extract_txt(filepath)
    else:
        raise ValueError(f"不支持的文件格式: {filepath}\n支持的格式: PDF, DOCX, TXT, MD")

    # 后处理
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text


def main():
    parser = argparse.ArgumentParser(description="论文文本提取工具")
    parser.add_argument("filepath", help="论文文件路径 (PDF/DOCX/TXT/MD)")
    parser.add_argument("--output", "-o", help="输出文件路径（可选，默认输出到同名.txt文件）")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不打印到stdout")
    args = parser.parse_args()

    try:
        text = extract(args.filepath)

        # 确定输出路径
        if args.output:
            out_path = args.output
        else:
            base = os.path.splitext(args.filepath)[0]
            out_path = f"{base}_extracted.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        if not args.quiet:
            print(f"✅ 提取完成: {args.filepath}")
            print(f"📄 输出文件: {out_path}")
            print(f"📏 字符数: {len(text):,}")
            # 打印前500字预览
            print(f"\n--- 文本预览（前500字）---")
            print(text[:500])

    except Exception as e:
        print(f"❌ 提取失败: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
