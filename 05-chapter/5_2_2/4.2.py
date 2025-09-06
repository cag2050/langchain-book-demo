from langchain_text_splitters import LatexTextSplitter

latex_text = "### 第一章 #### 第一节 详细内容"
markdown_splitter = LatexTextSplitter(
    chunk_size=5,
    chunk_overlap=1,
)
docs = markdown_splitter.create_documents([latex_text])
print(docs)
