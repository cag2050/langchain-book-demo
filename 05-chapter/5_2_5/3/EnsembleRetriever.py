from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv()

doc_list = [
    "我喜欢苹果",
    "我喜欢橙子",
    "苹果和橙子都是水果",
]

# 初始化BM25检索器
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 1

# 使用阿里云通用文本向量模型初始化Chroma检索器
embedding = DashScopeEmbeddings()
chroma_vectorstore = Chroma.from_texts(doc_list, embedding=embedding)
chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 1})

# 初始化EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, chroma_retriever],
    weights=[0.5, 0.5],
)

docs = ensemble_retriever.invoke("苹果")
print(docs)
