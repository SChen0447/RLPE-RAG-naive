from app.config import K


def build_retriever(vectordb, k = K):
    return vectordb.as_retriever(search_kwargs={"k": k})

