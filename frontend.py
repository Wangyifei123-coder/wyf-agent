"""WYF Agent — Streamlit 前端对话界面"""

from __future__ import annotations

import base64
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8080")

st.set_page_config(
    page_title="WYF Agent",
    page_icon=":robot:",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* ===== 全局布局 ===== */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    .stApp > header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.5rem 0;
    }

    /* ===== 登录页面 ===== */
    [data-testid="stForm"] {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
    }
    [data-testid="stForm"] h1, [data-testid="stForm"] h2 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    [data-testid="stForm"] button[type="submit"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    [data-testid="stForm"] button[type="submit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }

    /* ===== 侧边栏 ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9ff 0%, #f0f2ff 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: #667eea;
        font-weight: 600;
    }
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }

    /* ===== 意图标签 ===== */
    .intent-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .intent-tag:hover { transform: scale(1.05); }
    .intent-chitchat {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    .intent-knowledge_qa {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565c0;
        border: 1px solid #90caf9;
    }
    .intent-doc_analysis {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        color: #e65100;
        border: 1px solid #ffcc80;
    }

    /* ===== 引用来源 ===== */
    .source-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        background: linear-gradient(135deg, #f5f5f5 0%, #eeeeee 100%);
        color: #424242;
        font-size: 11px;
        font-weight: 500;
        margin: 3px 4px 3px 0;
        border: 1px solid #e0e0e0;
        transition: all 0.2s ease;
    }
    .source-tag:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: transparent;
        transform: translateY(-1px);
    }

    /* ===== 消息气泡 ===== */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    [data-testid="stChatMessage"]:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    [data-testid="stChatMessage"][data-testid-type="user"] {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
    }
    [data-testid="stChatMessage"][data-testid-type="assistant"] {
        background: white;
        border-left: 4px solid #764ba2;
    }

    /* ===== 图片样式 ===== */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid white;
    }
    [data-testid="stImage"]:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    [data-testid="stImage"] img {
        border-radius: 10px;
    }

    /* ===== 聊天输入框 ===== */
    [data-testid="stChatInput"] {
        border-radius: 24px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1rem;
        background: white;
        transition: all 0.3s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }

    /* ===== 按钮通用样式 ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* ===== 状态提示 ===== */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* ===== 分割线 ===== */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* ===== 加载动画 ===== */
    .loading-spinner {
        display: inline-block;
        width: 16px;
        height: 16px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s ease-in-out infinite;
        margin-right: 8px;
        vertical-align: middle;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* ===== 计时器显示 ===== */
    .timer-display {
        font-size: 12px;
        color: #667eea;
        padding: 4px 12px;
        background: linear-gradient(
            135deg,
            rgba(102, 126, 234, 0.08) 0%,
            rgba(118, 75, 162, 0.08) 100%
        );
        border-radius: 20px;
        display: inline-block;
        margin-top: 8px;
        font-weight: 500;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    /* ===== 标题样式 ===== */
    h1, h2, h3 {
        color: #1a1a2e;
        font-weight: 700;
    }

    /* ===== 隐藏 Streamlit 默认元素 ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ===== 动画效果 ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
        animation: fadeIn 0.3s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_api(
    endpoint: str,
    method: str = "GET",
    json_data: dict | None = None,
    timeout: int = 120,
    token: str | None = None,
) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "POST":
                resp = client.post(
                    f"{API_BASE}{endpoint}",
                    json=json_data,
                    headers=headers,
                )
            else:
                resp = client.get(f"{API_BASE}{endpoint}", headers=headers)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("无法连接到后端服务")
        st.stop()
    except Exception as e:
        st.error(f"请求失败: {e}")
        return {}


if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.token:
    st.title("WYF Agent 登录")
    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True)
        if submitted:
            result = call_api(
                "/auth/login",
                "POST",
                {"username": username, "password": password},
            )
            if result.get("success"):
                st.session_state.token = result["token"]
                st.session_state.username = result["username"]
                st.rerun()
            else:
                st.error(result.get("error", "登录失败"))

    st.info("演示账号: admin/admin123 或 user/user123")
    st.stop()

with st.sidebar:
    st.title("WYF Agent")
    st.text(f"当前用户: {st.session_state.username}")
    if st.button("退出登录", use_container_width=True):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

    st.markdown("---")

    st.subheader("知识库状态")
    try:
        stats = call_api("/knowledge/stats", token=st.session_state.token)
        st.metric("文档数量", stats.get("count", 0))
    except Exception:
        st.warning("无法获取知识库状态")

    st.markdown("---")

    st.subheader("文档入库")
    uploaded_files = st.file_uploader(
        "上传文档到知识库",
        type=["md", "txt", "pdf", "docx", "xlsx", "csv"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("开始入库", use_container_width=True):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in uploaded_files:
                filepath = os.path.join(tmpdir, f.name)
                with open(filepath, "wb") as out:
                    out.write(f.getvalue())
            with st.spinner("正在入库..."):
                result = call_api(
                    "/knowledge/ingest",
                    "POST",
                    {"path": tmpdir},
                    token=st.session_state.token,
                )
                st.success(
                    f"入库完成: {result.get('documents_loaded', 0)} 篇文档, "
                    f"{result.get('chunks_created', 0)} 个 chunk"
                )

    st.markdown("---")

    st.subheader("图片上传")
    uploaded_images = st.file_uploader(
        "上传图片（可多选）",
        type=["png", "jpg", "jpeg", "gif", "webp"],
        accept_multiple_files=True,
    )

    st.markdown("---")

    st.subheader("系统信息")
    try:
        health = call_api("/health", token=st.session_state.token)
        st.text(f"状态: {health.get('status', 'unknown')}")
        st.text(f"版本: {health.get('version', 'unknown')}")
    except Exception:
        st.text("状态: 未连接")

    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("WYF Agent 对话")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_images" not in st.session_state:
    st.session_state.pending_images = []

if uploaded_images:
    st.session_state.pending_images = uploaded_images
    cols = st.columns(min(len(uploaded_images), 4))
    for i, img in enumerate(uploaded_images):
        with cols[i % 4]:
            st.image(img, width=200, caption=img.name)

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    intent = msg.get("intent", "")
    sources = msg.get("sources", [])

    with st.chat_message(role):
        if msg.get("images"):
            cols = st.columns(min(len(msg["images"]), 4))
            for i, img_b64 in enumerate(msg["images"]):
                with cols[i % 4]:
                    st.image(base64.b64decode(img_b64), width=200)
        st.markdown(content)
        if intent and role == "assistant":
            intent_class = f"intent-{intent}"
            intent_label = {
                "chitchat": "闲聊",
                "knowledge_qa": "知识问答",
                "doc_analysis": "文档分析",
            }.get(intent, intent)
            st.markdown(
                f'<span class="intent-tag {intent_class}">{intent_label}</span>',
                unsafe_allow_html=True,
            )
        if sources and role == "assistant":
            source_html = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
            st.markdown(f"**引用来源:** {source_html}", unsafe_allow_html=True)

if prompt := st.chat_input("输入你的问题或网页URL..."):
    import re

    url_pattern = re.compile(r'https?://\S+')
    urls = url_pattern.findall(prompt)

    image_b64_list = []
    for img in st.session_state.pending_images:
        image_b64_list.append(base64.b64encode(img.getvalue()).decode())

    if urls and not image_b64_list:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            results = []

            for url in urls:
                placeholder.markdown(f"正在入库网页: {url} ...")
                try:
                    result = call_api(
                        "/knowledge/ingest-url", "POST",
                        {"url": url}, timeout=180,
                        token=st.session_state.token,
                    )
                    results.append(result)
                    placeholder.markdown(
                        f"网页入库完成: **{result.get('title', '')}**\n"
                        f"- 文字分块: {result.get('text_chunks', 0)}\n"
                        f"- 图片提取: {result.get('images_extracted', 0)}"
                    )
                except Exception as e:
                    placeholder.markdown(f"网页入库失败: {e}")

            full_answer = "网页入库完成！你现在可以问我关于这些网页内容的问题了。"
            placeholder.markdown(full_answer)
            intent = "doc_analysis"
            sources = urls

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer,
            "intent": intent,
            "sources": sources,
        })
        st.session_state.pending_images = []
        st.rerun()
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
                "images": image_b64_list,
            }
        )

        with st.chat_message("user"):
            if image_b64_list:
                cols = st.columns(min(len(image_b64_list), 4))
                for i, img_b64 in enumerate(image_b64_list):
                    with cols[i % 4]:
                        st.image(base64.b64decode(img_b64), width=200)
            st.markdown(prompt)

        with st.chat_message("assistant"):
            import json as _json
            import time

            placeholder = st.empty()
            timer_placeholder = st.empty()
            full_answer = ""
            intent = "knowledge_qa"
            sources: list[str] = []

            start_time = time.time()

            def update_timer():
                elapsed = time.time() - start_time
                timer_placeholder.markdown(
                    f'<div class="timer-display">'
                    f'<span class="loading-spinner"></span>'
                    f'思考中... {elapsed:.1f}s</div>',
                    unsafe_allow_html=True,
                )

            update_timer()

            try:
                with httpx.Client(timeout=180) as client:
                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                    with client.stream(
                        "POST",
                        f"{API_BASE}/chat/stream",
                        json={
                            "message": prompt,
                            "images": image_b64_list or None,
                        },
                        headers=headers,
                    ) as response:
                        for line in response.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            data = _json.loads(line[6:])
                            msg_type = data.get("type", "")
                            if msg_type == "intent":
                                intent = data.get("intent", intent)
                            elif msg_type == "chunk":
                                chunk = data.get("chunk", "")
                                full_answer += chunk
                                update_timer()
                                placeholder.markdown(full_answer)
                            elif msg_type == "done":
                                intent = data.get("intent", intent)
                                sources = data.get("sources", [])
                                break
            except Exception as e:
                st.error(f"请求失败: {e}")

            elapsed = time.time() - start_time
            timer_placeholder.markdown(
                f'<div class="timer-display">'
                f'回答完成 {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

            if full_answer:
                placeholder.markdown(full_answer)

                intent_class = f"intent-{intent}"
                intent_label = {
                    "chitchat": "闲聊",
                    "knowledge_qa": "知识问答",
                    "doc_analysis": "文档分析",
                }.get(intent, intent)
                st.markdown(
                    f'<span class="intent-tag {intent_class}">{intent_label}</span>',
                    unsafe_allow_html=True,
                )

                if sources:
                    source_html = "".join(f'<span class="source-tag">{s}</span>' for s in sources)
                    st.markdown(f"**引用来源:** {source_html}", unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_answer,
                "intent": intent,
                "sources": sources,
            }
        )
        st.session_state.pending_images = []
