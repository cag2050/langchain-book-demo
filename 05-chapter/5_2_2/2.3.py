from langchain_text_splitters import SpacyTextSplitter

text = "自然语言处理(NLP)是人工智能的核心领域之一。\n它研究如何让计算机理解人类语言。"
# python -m spacy download zh_core_web_sm -i https://pypi.tuna.tsinghua.edu.cn/simple
text_splitter = SpacyTextSplitter(pipeline="zh_core_web_sm")
docs = text_splitter.split_text(text)
print(docs)
