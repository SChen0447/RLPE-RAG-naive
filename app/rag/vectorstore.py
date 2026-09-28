from app.rag.embeddings import BGEEmbeddings

from pathlib import Path

from langchain_chroma import Chroma

from app.config import EMBEDDING_MODEL, VECTOR_DB_PATH, K


class ChromaVectorStore:
    def __init__(
            self, 
            persist_directory: str = VECTOR_DB_PATH,
            embedding_model: str = EMBEDDING_MODEL
    ):
        self.persist_directory = persist_directory
        self.embeddings = BGEEmbeddings(model_name=embedding_model)
        self._vectordb: Chroma|None = None


    def build(self, chunks, force_rebuild: bool = False):
        """构建或重建数据库"""
        persist_path = Path(self.persist_directory)
        if force_rebuild and persist_path.exists():
            import shutil
            shutil.rmtree(persist_path)

        self._vectordb = Chroma.from_documents(
            documents = chunks,
            embedding = self.embeddings,
            persist_directory=self.persist_directory
        )
        

    def load(self):
        """加载数据库"""
        self._vectordb = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )


    def search(self, query: str, k: int = K):
        """检索"""
        if self._vectordb is None:
            raise ValueError("向量数据库未加载，请先调用 load() 方法加载数据库。")
        return self._vectordb.similarity_search(query, k=k)

    def get_vectorstore(self):
        return self._vectordb

    


# 面向过程
# file_paths = dp.get_file_paths()
# documents = dp.load_documents(file_paths)
# chunks = dp.split_documents(documents)

# persist_directory="data_base/vector_db/chroma"

# vectordb = Chroma.from_documents(
#     documents = chunks,
#     embedding = BGEEmbeddings(model_name="BAAI/bge-small-zh-v1.5"),
#     persist_directory=persist_directory
# )



if __name__ == "__main__":

    question = "什么是大语言模型？"
    vectordb = ChromaVectorStore()
    vectordb.load()
    results = vectordb.similarity_search(question, k=3)

    for i, sim_doc in enumerate(results):
        print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

