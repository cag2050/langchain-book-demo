from langchain.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"

load_dotenv()

loader = WebBaseLoader("https://www.ituring.com.cn/book/3457")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=0,
)
splits = text_splitter.split_documents(data)

embedding = DashScopeEmbeddings()
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)

question = "介绍一下《LangChain编程：从入门到实践（第2版）》这本书"

llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
# 使用多查询检索器，结合向量数据库和语言模型
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(),
    llm=llm,
)

# 在源代码：langchain-book-demo/venv/lib/python3.11/site-packages/langchain/retrievers/multi_query.py的def _get_relevant_documents函数中，添加行：print(queries)，可以看到question问题被自动转换成3个不同的查询意图。
unique_docs = retriever_from_llm.invoke(question)
print(unique_docs)
# Todo 输出的内容，不是介绍书的，不清楚哪里有问题。
# 输出内容：[Document(metadata={'description': '图灵社区成立于2005年6月，以策划出版高质量的科技书籍为核心业务，主要出版领域包括计算机、电子电气、数学统计、科普等，通过引进国际高水平的教材、专著，以及发掘国内优秀原创作品等途径，为目标读者提供一流的内容。', 'title': '图灵社区', 'source': 'https://www.ituring.com.cn/book/3457', 'language': 'en'}, page_content="图灵社区We're sorry but ituring-mobile3 doesn't work properly without JavaScript enabled. Please enable it to continue.")]
