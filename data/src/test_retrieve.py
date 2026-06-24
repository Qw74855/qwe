# src/test_retrieve.py
import os
# 统一配置HuggingFace国内镜像，和build_vector_db保持一致
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 初始化与建库完全相同的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

# 2. 加载本地已持久化的向量库
vector_db = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)

# 3. 实训文档原版检索函数
def search_knowledge(query):
    # 检索相似度前3条结果
    results = vector_db.similarity_search(query, k=3)
    for r in results:
        print(f"相关元数据: {r.metadata}")
        print(f"匹配内容: {r.page_content}\n")
    return results

# 4. 测试入口，实训示例问题
if __name__ == "__main__":
    print("=====测试查询：我发烧了怎么办?=====")
    search_knowledge("我发烧了怎么办?")

    # 额外两个检查点测试用例
    print("=====测试查询：奖学金需要什么条件？=====")
    search_knowledge("奖学金需要什么条件？")

    print("=====测试查询：宿舍报修怎么操作？=====")
    search_knowledge("宿舍报修怎么操作？")