"""WYF Agent — Streamlit 前端对话界面"""

from __future__ import annotations

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
    .stApp { max-width: 1200px; margin: 0 auto; }
    .intent-tag {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 12px; font-weight: bold; margin-left: 8px;
    }
    .intent-chitchat { background: #e8f5e9; color: #2e7d32; }
    .intent-knowledge_qa { background: #e3f2fd; color: #1565c0; }
    .intent-doc_analysis { background: #fff3e0; color: #e65100; }
    .source-tag {
        display: inline-block; padding: 2px 6px; border-radius: 3px;
        background: #f5f5f5; color: #616161; font-size: 11px;
        margin: 2px 4px 2px 0;
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
                    f"{API_BASE}{endpoint}", json=json_data, headers=headers,
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
                "/auth/login", "POST",
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
                    "/knowledge/ingest", "POST",
                    {"path": tmpdir}, token=st.session_state.token,
                )
                st.success(
                    f"入库完成: {result.get('documents_loaded', 0)} 篇文档, "
                    f"{result.get('chunks_created', 0)} 个 chunk"
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

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    intent = msg.get("intent", "")
    sources = msg.get("sources", [])

    with st.chat_message(role):
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
            source_html = "".join(
                f'<span class="source-tag">{s}</span>' for s in sources
            )
            st.markdown(f"**引用来源:** {source_html}", unsafe_allow_html=True)

if prompt := st.chat_input("输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        import json as _json

        placeholder = st.empty()
        full_answer = ""
        intent = "knowledge_qa"
        sources: list[str] = []

        try:
            with httpx.Client(timeout=180) as client:
                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                with client.stream(
                    "POST",
                    f"{API_BASE}/chat/stream",
                    json={"message": prompt},
                    headers=headers,
                ) as response:
                    for line in response.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = _json.loads(line[6:])
                        if data.get("error"):
                            st.error(data["error"])
                            break
                        if data.get("done"):
                            intent = data.get("intent", intent)
                            sources = data.get("sources", [])
                            break
                        chunk = data.get("chunk", "")
                        full_answer += chunk
                        placeholder.markdown(full_answer)
        except Exception as e:
            st.error(f"请求失败: {e}")

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
                source_html = "".join(
                    f'<span class="source-tag">{s}</span>' for s in sources
                )
                st.markdown(f"**引用来源:** {source_html}", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "intent": intent,
        "sources": sources,
    })
