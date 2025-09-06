from langchain_text_splitters import NLTKTextSplitter

text = "自然语言处理(NLP)是人工智能的核心领域之一。\n它研究如何让计算机理解人类语言。"
text_splitter = NLTKTextSplitter()
docs = text_splitter.split_text(text)
print(docs)
