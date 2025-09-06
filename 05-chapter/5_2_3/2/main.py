from dotenv import load_dotenv

from langchain_community.embeddings.dashscope import DashScopeEmbeddings

load_dotenv()


def test_embeddings():
    embeddings_model = DashScopeEmbeddings()
    embeddings = embeddings_model.embed_documents(
        [
            "《星际穿越》：这是一部探讨宇宙奥秘，讲述宇航员穿越虫洞寻找人类新家园的故事的科幻电影",
            "《阿甘正传》：这部励志电影讲述了一位智力有限但心灵纯净的男子，他意外的参与了多个重大历史事件",
            "《泰坦尼克号》：一部讲述了1912年泰坦尼克号沉船事故中，两位来自不同阶层的年轻人之间爱情故事的浪漫电影"
        ]
    )
    embedded_query = embeddings_model.embed_query("我想看一部关于宇宙探险的电影")
    # 每个句子被嵌入为1536维的向量（长度为1536的浮点数数组）
    print(len(embeddings), len(embeddings[0]), len(embedded_query))


if __name__ == '__main__':
    test_embeddings()
