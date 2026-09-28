from app.rag.vectorstore import ChromaVectorStore
from app.rag.retriever import build_retriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.llm import get_llm


# 没有聊天记录功能
# def format_docs(docs: str) -> str:
#     """把检索到的 Document 列表拼成上下文字符串。"""
#     if not docs:
#         return "（没有检索到相关资料）"
#     return "\n\n".join(doc.page_content for doc in docs)


# def build_retrieval_chain(vectordb: ChromaVectorStore):
#     # 加载数据库放在main里面
#     # vectordb = ChromaVectorStore()
#     # vectordb.load()
#     retriever = build_retriever(vectordb.get_vectorstore())
#     formater = RunnableLambda(format_docs)

#     retrieval_chain = retriever | formater
#     return retrieval_chain


# template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
# 案。最多使用三句话。尽量使答案简明扼要。请你在回答的最后说“谢谢你的提问！”。
# {context}
# 问题: {input}
# """


# def build_qa_chain():
#     retrieval_chain = build_retrieval_chain()
#     prompt = PromptTemplate(template=template)
#     llm = get_llm()
#     qa_chain = (
#         RunnableParallel({"context": retrieval_chain, "input": RunnablePassthrough()})
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return qa_chain



# 添加聊天记录功能
def format_docs(docs: dict):
    """把检索到的 Document 列表拼成上下文字符串。"""
    if not docs:
        return "（没有检索到相关资料）"
    return "\n\n".join(doc.page_content for doc in docs["context"])



def build_qa_history_chain(vectordb: ChromaVectorStore):
    retriever = build_retriever(vectordb.get_vectorstore())
    llm = get_llm()


    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题,如果没有多余的聊天记录则返回用户的问题"
    )
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    retrieval_chain = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever),
        condense_question_prompt | llm | StrOutputParser() | retriever
    )


    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = (
        RunnablePassthrough().assign(context=format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieval_chain,
    ).assign(answer=qa_chain)

    return qa_history_chain
















