from langchain_core.runnables import RunnableLambda


def test():
    # 使用|运算符构造的RunnableSequence
    sequence1 = RunnableLambda(lambda x: x - 1) | RunnableLambda(lambda x: x * 2)
    print(sequence1.invoke(3))
    print(sequence1.batch([1, 2, 3]))

    # 包含使用字典字面值构造的RunnableParallel的序列
    sequence2 = RunnableLambda(lambda x: x * 2) | {
        'sub_1': RunnableLambda(lambda x: x - 1),
        'sub_2': RunnableLambda(lambda x: x - 2),
    }
    print(sequence2.invoke(3))


if __name__ == "__main__":
    test()
