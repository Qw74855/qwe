import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import streamlit as st
import os
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa
# 加载环境变量
load_dotenv()
# ------------------- 页面基础配置 -------------------
st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)
# 全局美化CSS
st.markdown("""
<style>
[data-testid="stChatMessage"] {
    max-width: 85%;
    margin-left: auto;
    margin-right: auto;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 10px;
}
.stButton > button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)
# ------------------- 缓存资源 -------------------
@st.cache_resource(show_spinner="正在加载文本向量化模型...")
def load_embeddings():    
return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True}    
)
@st.cache_resource(show_spinner="正在加载校园知识库向量库...")
def load_vector_db():
    embeddings = load_embeddings()
    db_path = "./vector_db"    
if not os.path.exists(db_path):
        st.warning(f"向量库目录 {db_path} 不存在，请先导入校园文档生成知识库！")    
return Chroma(persist_directory=db_path, embedding_function=embeddings)
embeddings = load_embeddings()
vector_db = load_vector_db()
# 校验星火API密钥
APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 环境变量缺失！请在项目根目录 .env 文件中配置 SPARK_APIPASSWORD 星火接口密钥")
    st.stop()
# ------------------- RAG问答核心函数 -------------------
def rag_retrieve_answer(question):    
try:
        docs = vector_db.similarity_search(question, k=3)        
if len(docs) == 0:            
return "📭 知识库未查询到相关内容，暂时无法解答该问题，你可以咨询教务处相关老师。"        
        context = "\n\n=====分割线=====\n\n".join([doc.page_content.strip() for doc in docs])
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
        headers = {            
"Content-Type": "application/json",            
"Authorization": f"Bearer {APIPASSWORD}"        
}
        payload = {            
"model": "spark-x",            
"messages": [{"role": "user", "content": prompt_text}],            
"temperature": 0.3,            
"max_tokens": 1024        
}
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
