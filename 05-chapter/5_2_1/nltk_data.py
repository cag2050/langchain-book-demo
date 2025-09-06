import nltk
from nltk.tokenize import word_tokenize
import jieba
from nltk import Text

print(nltk.data.path)

# 下载 https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt_tab.zip ，
# 放到/Users/cag2050/nltk_data/tokenizers目录下并解压punkt_tab.zip（目录：/Users/cag2050/nltk_data，是nltk.data.path输出的其中一个目录，也可以放到nltk.data.path输出的其它目录）
print(word_tokenize("This is a test sentence."))

# 中文分词处理
tokens = list(jieba.cut("自然语言处理很重要"))
text = Text(tokens)  # 转为NLTK可处理对象
print(text)
