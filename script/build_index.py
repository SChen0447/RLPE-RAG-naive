from app.rag.loaders import get_file_paths, load_documents, split_documents
from app.rag.embeddings import BGEEmbeddings
from app.rag.vectorstore import ChromaVectorStore
from app.llm import get_llm


file_paths = get_file_paths()
documents = load_documents(file_paths)
chunks = split_documents(documents)

vectorstore = ChromaVectorStore()
vectorstore.build(chunks)

vectordb = vectorstore.get_vectorstore()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})


question = "什么是prompt engineering?"

result = retriever.invoke(question)

for i, doc in enumerate(result):
    print(f"检索到的第{i}个内容: \n {doc.page_content}", end="\n-----------------------------------------------------\n")











