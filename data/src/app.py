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
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ 大模型接口调用失败\n状态码：{resp.status_code}\n返回信息：{resp.text[:300]}"

    except requests.exceptions.Timeout:
        return "⏱️ 请求超时，服务器响应缓慢，请重新提问！"
    except Exception as e:
        return f"⚠️ 问答流程出现未知异常：{str(e)}"

# ------------------- 意图路由函数 -------------------
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

# ------------------- 精简侧边栏 -------------------
with st.sidebar:
    st.header("🏫 校园百事通")
    st.divider()
    # 快捷提问缓存
    if "temp_input" not in st.session_state:
        st.session_state["temp_input"] = ""
    # 清空对话按钮
    if st.button("🗑️ 清空全部对话记录", type="secondary"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.info("切换上方标签页使用不同功能")

# ====================== 顶部Tab标签布局核心改动 ======================
st.title("校园生活百事通助手")
st.markdown("基于本地知识库RAG大模型，一站式解决校园各类问题")
st.divider()

# 创建三个标签页
tab_chat, tab_tool, tab_intro = st.tabs(["💬 智能对话", "🧮 工具速算", "📖 功能说明"])

# ---------------- Tab1：智能对话（滚动聊天框布局） ----------------
with tab_chat:
    # 初始化对话记录
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "你好！我是校园百事通，有任何校园问题、想查教学周、计算绩点都可以直接问我~"}
        ]

    # 快捷提问示例按钮
    sample_q = [
        "现在是第几教学周？",
        "帮我计算绩点 90,82,75,60",
        "学校奖学金申请条件是什么？"
    ]
    btn_cols = st.columns(3)
    for idx, q in enumerate(sample_q):
        with btn_cols[idx]:
            if st.button(q):
                st.session_state["temp_input"] = q

    st.divider()
    # 固定高度聊天窗口
    chat_container = st.container(height=550, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            avatar = "👤" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

    st.divider()
    # 底部输入框
    input_text = st.chat_input("请输入你的校园问题...")
    # 侧边快捷按钮赋值
    if st.session_state["temp_input"]:
        input_text = st.session_state["temp_input"]
        st.session_state["temp_input"] = ""

    # 处理提问逻辑
    if input_text:
        st.session_state.messages.append({"role": "user", "content": input_text})
        st.rerun()

    if input_text:
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI正在检索知识库并思考答案..."):
                    res = agent_answer(input_text)
                st.markdown(res)
            st.session_state.messages.append({"role": "assistant", "content": res})

# ---------------- Tab2：工具速算（独立绩点/周数工具面板） ----------------
with tab_tool:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("📅 教学周查询")
        if st.button("一键查询当前教学周"):
            week_res = get_current_week()
            st.success(week_res)
    with col2:
        st.subheader("📊 GPA绩点计算")
        score_input = st.text_input("输入成绩，逗号分隔", placeholder="例如：85,92,78,66")
        if st.button("计算平均绩点") and score_input:
            nums = re.findall(r'\d+', score_input)
            if nums:
                gpa_res = calculate_gpa(','.join(nums))
                st.success(gpa_res)
            else:
                st.warning("未识别到有效分数，请重新输入")

# ---------------- Tab3：功能说明 ----------------
with tab_intro:
    st.subheader("✨ 三大核心能力")
    st.markdown("""
    1. **📚 校园知识库问答**
    覆盖校规、宿舍、选课、奖学金、社团等校内规章制度，基于本地文档检索精准回答
    
    2. **📅 教学周自动查询**
    一键获取当前学期教学周，无需手动对照校历
    
    3. **📊 百分制GPA绩点换算**
    批量录入多门课程分数，自动计算平均绩点
    """)
    st.divider()
    st.subheader("使用提示")
    st.markdown("""
    - 咨询校园相关政策直接在【智能对话】提问
    - 快速算分、查周数切换至【工具速算】标签页使用独立面板
    - 对话记录可在左侧边栏一键清空
    """)
