# src/build_vector_db.py
import os
# 统一国内镜像，和检索脚本一致
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, DirectoryLoader

# 1. 初始化嵌入模型（和检索代码完全一致）
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh"
)

# 2. 加载校园知识库文档（存放校园规则、奖学金、报修等txt/md文件）
# 把你的所有校园文档放在 ./docs 文件夹下
loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()

# 3. 文本分块
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
split_docs = text_splitter.split_documents(documents)

# 4. 构建并持久化向量库到 ./vector_db
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./vector_db"
)
# 持久化保存到本地文件夹
vector_db.persist()
print("向量库构建完成，已保存至 ./vector_db")
