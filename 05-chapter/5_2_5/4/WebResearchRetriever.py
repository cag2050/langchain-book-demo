from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers.web_research import WebResearchRetriever
from langchain_community.utilities import GoogleSearchAPIWrapper
# from langchain_google_community import GoogleSearchAPIWrapper
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv()

vectorstore = Chroma(
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma_db_oai"
)

llm = ChatDeepSeek(model="deepseek-chat")

search = GoogleSearchAPIWrapper()

web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
    allow_dangerous_requests=True
)

user_input = "LLM驱动的自主代理是如何工作的？"
docs = web_research_retriever.invoke(user_input)
'''
TODO 报错：
  File "/Users/cag2050/Documents/PycharmProjects/langchain-book-demo/venv/lib/python3.11/site-packages/httplib2/__init__.py", line 1154, in connect
    sock.connect((self.host, self.port))
TimeoutError: timed out
未知原因，一个可能的解决办法：若遇网络限制，可通过代理服务（如 http://api.wlai.vip）提升稳定性，但需确保代理同时支持密钥和 CSE ID 的传递。
'''
print(docs)
