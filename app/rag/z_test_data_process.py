from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re


# 加载PDF文档
pdf_loader = PyMuPDFLoader("data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")
pdf_pages = pdf_loader.load()

print(f"Type: {type(pdf_pages)}, Length: {len(pdf_pages)}页")

print(f"每一个元素的类型：{type(pdf_pages[1])}.", 
    f"该文档的描述性数据：{pdf_pages[1].metadata}", 
    f"查看该文档的内容:\n{pdf_pages[1].page_content}", 
    sep="\n------\n")



# 加载Markdown文档
md_loader = UnstructuredMarkdownLoader("data_base/knowledge_db/prompt_engineering/1. 简介 Introduction.md")
md_pages = md_loader.load()

print(f"Type: {type(md_pages)}, Length: {len(md_pages)}页")

print(f"每一个元素的类型：{type(md_pages[0])}.", 
    f"该文档的描述性数据：{md_pages[0].metadata}", 
    f"查看该文档的内容:\n{md_pages[0].page_content}", 
    sep="\n------\n")



# 数据处理
pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)

for pdf_page in pdf_pages:
    pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
    pdf_page.page_content = pdf_page.page_content.replace('•', '')
    pdf_page.page_content = pdf_page.page_content.replace(' ', '')

for md_page in md_pages:
    md_page.page_content = md_page.page_content.replace('\n\n', '\n')



# 文本分块
CHUNK_SIZE = 500
OVERLAP_SIZE = 50

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE,
)

pdf_chunks = text_splitter.split_documents(pdf_pages)
md_chunks = text_splitter.split_documents(md_pages)


