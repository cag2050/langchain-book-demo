from langgraph.store.memory import InMemoryStore

# 长期记忆
# InMemoryStore将数据保存到内存中
store = InMemoryStore()
user_id = "123"
app_context = "chat_context"
namespace = (user_id, app_context)

# 保存一个记忆
store.put(namespace, "memory", {"爱好": ["喜欢编程", "喜欢写作"], "职业": "程序员"})
# 通过key获取记忆
item = store.get(namespace, "memory")
print(item)
# 搜索该命名空间下的所有记忆，并根据内容过滤
items = store.search(namespace, filter={"职业": "程序员"})
print(items)
