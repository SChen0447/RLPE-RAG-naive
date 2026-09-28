from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re


def load_pdf_document(pdf_path):
    pdf_loader = PyMuPDFLoader(pdf_path)
    pdf_pages = pdf_loader.load()
    return pdf_pages

def load_markdown_document(md_path):
    md_loader = UnstructuredMarkdownLoader(md_path)
    md_pages = md_loader.load()
    return md_pages


def process_pdf_pages(pdf_pages):
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    for pdf_page in pdf_pages:
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
        pdf_page.page_content = pdf_page.page_content.replace('•', '')
        pdf_page.page_content = pdf_page.page_content.replace(' ', '')
    return pdf_pages

def process_markdown_pages(md_pages):
    for md_page in md_pages:
        md_page.page_content = md_page.page_content.replace('\n\n', '\n')
    return md_pages


def split_documents(pages, chunk_size=500, overlap_size=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap_size,
    )
    chunks = text_splitter.split_documents(pages)
    return chunks


def load_and_process_pdf(pdf_path):
    pdf_pages = load_pdf_document(pdf_path)
    processed_pdf_pages = process_pdf_pages(pdf_pages)
    pdf_chunks = split_documents(processed_pdf_pages)
    return pdf_chunks

def load_and_process_markdown(md_path):
    md_pages = load_markdown_document(md_path)
    processed_md_pages = process_markdown_pages(md_pages)
    md_chunks = split_documents(processed_md_pages)
    return md_chunks



