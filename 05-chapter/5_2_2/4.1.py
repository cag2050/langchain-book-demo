from langchain_text_splitters import MarkdownTextSplitter

markdown_text = "### 第一章 #### 第一节 详细内容"
markdown_splitter = MarkdownTextSplitter(
    chunk_size=5,
    chunk_overlap=1,
)
docs = markdown_splitter.create_documents([markdown_text])
print(docs)
