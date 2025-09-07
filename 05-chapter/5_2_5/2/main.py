from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def test():
    # loader = WebBaseLoader("https://www.ituring.com.cn/book/3457")
    loader = WebBaseLoader("https://blog.csdn.net/ahdfwcevnhrtds/article/details/143317707")
    # loader = WebBaseLoader("https://juejin.cn/post/7346009985791311922?searchId=202509072001169178DF299F15A6D64EB4")
    # loader = WebBaseLoader("http://mp.weixin.qq.com/s/Y0t8qrmU5y6H93N-Z9_efw")
    loader.requests_kwargs = {'verify': False}  # 忽略SSL验证错误
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,
    )
    splits = text_splitter.split_documents(data)

    llm = ChatDeepSeek(model="deepseek-chat")
    retriever = Chroma.from_documents(
        documents=splits,
        embedding=DashScopeEmbeddings(),
    ).as_retriever()
    question = "内容是什么？"

    # 未压缩时的查询结果
    docs1 = retriever.invoke(input=question)
    pretty_print_docs(docs1)

    # 创建链式提取器
    compressor = LLMChainExtractor.from_llm(llm)
    # 创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )
    # 压缩后的查询结果
    docs2 = compression_retriever.invoke(input=question)
    pretty_print_docs(docs2)

    # 创建嵌入向量过滤器
    embeddings_filter = EmbeddingsFilter(
        embeddings=DashScopeEmbeddings(),
        similarity_threshold=0.76,
    )
    # 使用过滤器创建上下文压缩检索器
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=retriever,
    )
    docs3 = compression_retriever.invoke(input=question)
    pretty_print_docs(docs3)

if __name__ == "__main__":
    test()
