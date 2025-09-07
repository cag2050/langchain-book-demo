from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("../test.txt")
documents = loader.load()

input = "LLMOps的含义是什么？"

text_splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
)
texts = text_splitter.split_documents(documents)

embeddings = DashScopeEmbeddings()

db = Chroma.from_documents(texts, embeddings)

# 使用默认的相似性搜索
retriever = db.as_retriever()
docs = retriever.invoke("LLMOps的含义是什么？")
print("默认相关性搜索结果：\n", docs)

# 使用最大边际相关性（MMR）搜索
retriever_mmr = db.as_retriever(search_type="mmr")
docs_mmr = retriever_mmr.invoke(input)
print("MMR搜索结果：\n", docs_mmr)

# 设置相似度分数阈值
retriever_similarity_threshold = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5},
)
docs_similarity_threshold = retriever_similarity_threshold.invoke(input)
print("相似度分数阈值搜索结果：\n", docs_similarity_threshold)

# 指定Top k搜索
retriever_topk = db.as_retriever(search_kwargs={"k": 1})
docs_topk = retriever_topk.invoke(input)
print("Top k搜索：\n", docs_topk)
