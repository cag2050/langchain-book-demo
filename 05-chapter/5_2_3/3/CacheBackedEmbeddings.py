import time
from dotenv import load_dotenv

load_dotenv()


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        run_time = end - start
        print(f"{func.__name__} took {run_time} seconds.")
        return result

    return wrapper


@timing_decorator
def test_cache():
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

    underlying_embeddings = DashScopeEmbeddings()
    fs = LocalFileStore("./cache/")
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, fs, namespace=underlying_embeddings.model
    )
    embeddings = cached_embedder.embed_documents([
        "《星际穿越》：这是一部探讨宇宙奥秘，讲述宇航员穿越虫洞寻找人类新家园的故事的科幻电影",
        "《阿甘正传》：这部励志电影讲述了一位智力有限但心灵纯净的男子，他意外的参与了多个重大历史事件",
    ])


if __name__ == "__main__":
    # 第二次执行test_cache函数时，耗时会减少，因为直接从缓存中获取
    test_cache()
