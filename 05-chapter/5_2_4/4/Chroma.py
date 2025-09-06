from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)


def test_chromadb():
    raw_documents = TextLoader("./西游记.txt", encoding="utf-8").load()
    text_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=32,
    )
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, DashScopeEmbeddings())
    query = "孙悟空是怎么被压在五行山下的？"
    docs = db.similarity_search(query, k=1)
    print(docs[0].page_content)


if __name__ == "__main__":
    test_chromadb()
