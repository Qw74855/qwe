import streamlit as st
import re
import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

import os # HuggingFace国内镜像 os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
load_dotenv()

st.set_page_config(
    page_title="校园百事通",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------- 缓存资源加载（代码完全不变） -------------------
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

APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 环境变量缺失！请在项目根目录 .env 文件中配置 SPARK_APIPASSWORD 星火接口密钥")
    st.stop()

# ------------------- RAG、Agent函数完全不变 -------------------
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
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ 大模型接口调用失败\n状态码：{resp.status_code}\n返回信息：{resp.text[:300]}"
    except requests.exceptions.Timeout:
        return "⏱️ 请求超时，服务器响应缓慢，请重新提问！"
    except Exception as e:
        return f"⚠️ 问答流程出现未知异常：{str(e)}"

def agent_answer(question):
    week_pattern = r'第几周|第\d+周|本周|校历|现在几周|教学周'
    gpa_pattern = r'绩点|GPA|平均分|算分|成绩换算'
    if re.search(week_pattern, question):
        return get_current_week()
    if re.search(gpa_pattern, question):
        nums = re.findall(r'\d+', question)
        if nums:
            return calculate_gpa(','.join(nums))
        else:
            return """📝 绩点计算使用说明
请在问题中带上你的各科分数，示例：
帮我算绩点：88,76,92,65"""
    return rag_retrieve_answer(question)

# ====================== 全新布局：三栏并行 ======================
st.title("🏫 校园生活百事通助手")
st.markdown("一站式校园智能问答工具，支持知识库检索、教学周查询、GPA绩点计算")
st.divider()

# 三栏划分：左功能 | 中聊天 | 右帮助提示
col_left, col_mid, col_right = st.columns([0.8, 3, 1.2])

# 左栏：快捷功能控制
with col_left:
    st.subheader("⚡快捷操作")
    st.divider()
    sample_q = [
        "现在是第几教学周？",
        "帮我计算绩点 90,82,75,60",
        "学校奖学金申请条件是什么？",
        "宿舍灯坏了怎么办？",
    ]
    for q in sample_q:
        if st.button(q, use_container_width=True):
            st.session_state["temp_input"] = q
    st.divider()
    if st.button("🗑️ 清空对话", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 中间栏：核心聊天区域
with col_mid:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是校园百事通，有任何校园问题、想查教学周、计算绩点都可以直接问我~"}
        ]
    chat_box = st.container(height=550)
    with chat_box:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
                st.markdown(msg["content"])
    input_text = st.chat_input("输入你的校园问题...")
    if "temp_input" in st.session_state and st.session_state["temp_input"]:
        input_text = st.session_state["temp_input"]
        del st.session_state["temp_input"]
    if input_text:
        st.session_state.messages.append({"role": "user", "content": input_text})
        with chat_box:
            with st.chat_message("user", avatar="👤"):
                st.markdown(input_text)
        with chat_box:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI正在检索知识库并思考答案..."):
                    res = agent_answer(input_text)
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})

# 右栏：使用说明帮助面板
with col_right:
    st.subheader("📖 使用指南")
    st.divider()
    st.markdown("### 1. 知识库问答")
    st.caption("询问宿舍、选课、奖学金、社团、校规等校内政策")
    st.markdown("### 2. 教学周查询")
    st.caption("包含「本周、第几周、校历」等关键词自动识别")
    st.markdown("### 3. GPA绩点计算")
    st.caption("输入格式示例：帮我算绩点 85,92,77")
