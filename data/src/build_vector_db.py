# src/build_vector_db.py
import os
# 配置国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载csv问答数据
df = pd.read_csv('data/campus_data.csv')

# 2. 初始化本地中文嵌入模型（无需API，本地运行）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

# 3. 提取文本与元数据
texts = df['answer'].tolist()
metadatas = df[['id', 'category', 'question']].to_dict("records")

# 4. 创建并持久化Chroma向量数据库
vector_db = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./vector_db"
)
vector_db.persist()

# 打印入库结果
print(f"向量库构建完成，已存入{len(texts)}条问答记录")