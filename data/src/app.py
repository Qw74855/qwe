import streamlit as st
import os
import re
import requests
import base64
from dotenv import load_dotenv
from io import BytesIO

# -------------------------- 依赖&环境配置 --------------------------
# HuggingFace国内镜像，解决下载超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 向量库&嵌入模型导入
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 自定义工具与提示词
from prompt_templates import RAG_PROMPT
from tools import get_current_week, calculate_gpa

# -------------------------- 语音模块容错导入（sounddevice替代pyaudio，无pipwin） --------------------------
try:
    import speech_recognition as sr
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
    VOICE_INPUT_AVAILABLE = True
    SAMPLE_RATE = 16000
    RECORD_SECONDS = 3
except ModuleNotFoundError:
    VOICE_INPUT_AVAILABLE = False

try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False

# 加载环境变量
load_dotenv()

# -------------------------- 全局页面配置与美化CSS --------------------------
st.set_page_config(
    page_title="校园百事通AI助手",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义美化样式
custom_css = """
<style>
.main {background-color: #f7f9fc;}
.title-text {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    color: transparent;
    font-size: 32px !important;
    font-weight: bold;
    text-align: center;
    padding: 10px 0;
}
.stChatMessage {
    border-radius: 12px !important;
    padding: 8px 12px !important;
    margin: 6px 0 !important;
}
.sidebar-card {
    background: white;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 2px 10px #e2e8f0;
    margin-bottom: 15px;
}
.stButton>button {
    width: 100%;
    border-radius: 8px;
    background: #2563eb;
    color: white;
    font-weight: 500;
}
.stButton>button:hover {
    background: #1d4ed8;
}
.stChatInput {border-radius: 10px !important;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------- 缓存资源加载 --------------------------
@st.cache_resource(show_spinner="正在加载嵌入模型 BAAI/bge-small-zh ...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh",
        model_kwargs={"trust_remote_code": True}
    )

@st.cache_resource(show_spinner="正在加载向量知识库 Chroma ...")
def load_vector_db():
    embeddings = load_embeddings()
    if not os.path.exists("./vector_db"):
        st.warning("⚠️ 未检测到向量库文件夹，请先执行文档向量化脚本！")
        return None
    return Chroma(persist_directory="./vector_db", embedding_function=embeddings)

# 初始化资源
embeddings = load_embeddings()
vector_db = load_vector_db()

# 星火大模型密钥校验
APIPASSWORD = os.getenv("SPARK_APIPASSWORD")
if not APIPASSWORD:
    st.error("❌ 环境变量缺失：请在项目根目录 .env 文件配置 SPARK_APIPASSWORD 密钥")
    st.stop()

# -------------------------- 语音工具函数（sounddevice录音，无pyaudio） --------------------------
def voice_to_text():
    """麦克风录音转文字，完全抛弃pyaudio"""
    r = sr.Recognizer()
    st.info("🎤 正在收音，请说话...（3秒后自动停止）")
    # 录音
    audio_np = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    # 写入内存wav
    buf = BytesIO()
    sf.write(buf, audio_np, SAMPLE_RATE, format="wav")
    buf.seek(0)
    audio_file = sr.AudioFile(buf)
    with audio_file as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language="zh-CN")
        return text
    except sr.UnknownValueError:
        return "无法识别语音，请重新点击麦克风提问"
    except Exception as e:
        return f"语音识别失败：{str(e)}"

def text_to_voice(text):
    """文本转语音，返回音频播放器html"""
    if not TTS_AVAILABLE:
        return ""
    tts = gTTS(text=text, lang="zh-cn", slow=False)
    buf = BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    audio_bytes = buf.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
    <audio autoplay controls style="width:100%;margin-top:8px;">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    
    """
    return audio_html

# -------------------------- RAG、Agent业务核心函数 --------------------------
def rag_retrieve_answer(question):
    if vector_db is None:
        return "❌ 向量知识库未加载完成，无法检索校园资料"
    try:
        docs = vector_db.similarity_search(question, k=3)
        context = "\n\n".join([f"📄 参考文档片段：\n{d.page_content}" for d in docs])
        prompt_text = RAG_PROMPT.format(context=context, question=question)

        url = "https://spark-api-open.xf-yun.com/x2/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {APIPASSWORD}"
        }
        payload = {
            "model": "spark-x",
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.3
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            raw_ans = resp.json()["choices"][0]["message"]["content"]
            format_ans = raw_ans.replace("\n", "\n\n")
            return format_ans
        else:
            return f"❌ 星火API调用失败：状态码{resp.status_code}\n{resp.text}"
    except requests.exceptions.Timeout:
        return "⚠️ 大模型请求超时，请简化问题后重试"
    except Exception as e:
        return f"⚠️ 知识库检索异常：{str(e)}"

def agent_answer(question):
    # 校历周数路由
    if re.search(r'第.*周|校历|本周|几周|现在第几周', question):
        return get_current_week()
    # GPA绩点路由
    if re.search(r'绩点|GPA|平均分|算分', question):
        nums = re.findall(r'\d+', question)
        if nums:
            score_str = ','.join(nums)
            res = calculate_gpa(score_str)
            return f"📊 绩点计算结果：\n{res}"
        else:
            return "💡 请输入你的各科分数，示例：绩点计算 85,92,76"
    # 默认走RAG知识库问答
    return rag_retrieve_answer(question)

# -------------------------- 侧边栏UI --------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.subheader("⚙️ 助手设置面板")
    st.divider()
    # 语音播报开关
    voice_switch = st.toggle("开启回答语音播报", value=True if TTS_AVAILABLE else False, disabled=not TTS_AVAILABLE)
    st.divider()
    # 快捷提问模板
    st.subheader("📝 快捷提问模板")
    quick_q1 = st.button("现在是第几教学周？")
    quick_q2 = st.button("输入分数计算GPA绩点")
    quick_q3 = st.button("校园图书馆开放时间")
    st.divider()
    # 清空对话按钮
    if st.button("🗑️ 清空全部对话记录"):
        st.session_state.messages = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.subheader("💡 使用说明")
    tip_text = """
    1. 文本框输入校园问题
    2. 点击麦克风按钮语音提问（需安装sounddevice）
    3. 开启播报可朗读AI回答
    4. 支持校历查询 / GPA计算 / 校园知识库问答
    """
    if not VOICE_INPUT_AVAILABLE:
        tip_text += "\n⚠️ 当前未安装录音库，语音提问功能已禁用"
    if not TTS_AVAILABLE:
        tip_text += "\n⚠️ 当前未安装TTS库，语音播报功能已禁用"
    st.markdown(tip_text)
    st.markdown('</div>', unsafe_allow_html=True)

# 快捷提问绑定临时缓存
if quick_q1:
    st.session_state.temp_prompt = "现在是第几教学周？"
if quick_q2:
    st.session_state.temp_prompt = "输入分数计算GPA绩点"
if quick_q3:
    st.session_state.temp_prompt = "校园图书馆开放时间"

# -------------------------- 主页面聊天UI（修复低版本chat_input无value参数报错） --------------------------
st.markdown('<p class="title-text">🏫 校园生活百事通AI助手</p>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#64748b;margin-bottom:20px;">
    校园知识库问答 | 校历周数查询 | GPA绩点计算 | 语音对话助手
</div>
""", unsafe_allow_html=True)
st.divider()

# 初始化会话存储
if "messages" not in st.session_state:
    st.session_state.messages = []
if "temp_prompt" not in st.session_state:
    st.session_state.temp_prompt = ""

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"]=="user" else "🤖"):
        st.markdown(msg["content"])

# 底部输入区域
col_text, col_mic = st.columns([9, 1])
with col_text:
    # 移除不兼容的value参数，兼容所有streamlit版本
    input_val = st.chat_input("请输入你的校园问题...")

# 处理侧边快捷提问赋值
if st.session_state.temp_prompt != "":
    input_val = st.session_state.temp_prompt
    st.session_state.temp_prompt = ""

# 麦克风按钮仅在录音库可用时显示
mic_btn = False
if VOICE_INPUT_AVAILABLE:
    with col_mic:
        mic_btn = st.button("🎤")

# 分支1：麦克风语音提问
if mic_btn:
    voice_text = voice_to_text()
    if voice_text.startswith("无法识别") or voice_text.startswith("语音识别失败"):
        st.warning(voice_text)
    else:
        input_val = voice_text

# 分支2：文本提问处理逻辑
if input_val:
    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": input_val})
    with st.chat_message("user", avatar="👤"):
        st.markdown(input_val)

    # AI生成回答
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("AI正在检索校园资料并思考..."):
            ans = agent_answer(input_val)
        st.markdown(ans)
        # 语音播报
        if voice_switch and TTS_AVAILABLE:
            audio_html = text_to_voice(ans)
            st.markdown(audio_html, unsafe_allow_html=True)
    # 保存AI回复
    st.session_state.messages.append({"role": "assistant", "content": ans})
