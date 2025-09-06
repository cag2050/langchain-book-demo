from langchain_community.document_loaders import UnstructuredURLLoader
import logging
import ssl

logging.basicConfig(level=logging.DEBUG)


def test():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass  # Python 2.7兼容
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # 下载 https://github.com/nltk/nltk_data/blob/gh-pages/packages/tokenizers/punkt_tab.zip ，
    # 放到/Users/cag2050/nltk_data/tokenizers目录下并解压punkt_tab.zip（目录：/Users/cag2050/nltk_data，是nltk.data.path输出的其中一个目录，也可以放到nltk.data.path输出的其它目录）
    loader = UnstructuredURLLoader(
        urls=[
            "https://www.baidu.com",
        ],
        mode="elements",
        strategy="fast",
    )
    docs = loader.load()
    print(docs)


if __name__ == "__main__":
    test()
