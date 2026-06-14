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
    page_icon="https://img.icons8.com/color/48/robot-2.png",
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
) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "POST":
                resp = client.post(f"{API_BASE}{endpoint}", json=json_data)
            else:
                resp = client.get(f"{API_BASE}{endpoint}")
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        st.error("无法连接到后端服务，请确认服务已启动 (python -m uvicorn src.api:app --port 8080)")
        st.stop()
    except Exception as e:
        st.error(f"请求失败: {e}")
        return {}


with st.sidebar:
    st.title("WYF Agent")
    st.markdown("---")

    st.subheader("知识库状态")
    try:
        stats = call_api("/knowledge/stats")
        st.metric("文档数量", stats.get("count", 0))
    except Exception:
        st.warning("无法获取知识库状态")

    st.markdown("---")

    st.subheader("文档入库")
    uploaded_files = st.file_uploader(
        "上传文档到知识库",
        type=["md", "txt", "pdf"],
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
                result = call_api("/knowledge/ingest", "POST", {"path": tmpdir})
                st.success(
                    f"入库完成: {result.get('documents_loaded', 0)} 篇文档, "
                    f"{result.get('chunks_created', 0)} 个 chunk"
                )

    st.markdown("---")

    st.subheader("系统信息")
    try:
        health = call_api("/health")
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
        with st.spinner("思考中..."):
            response = call_api("/chat", "POST", {"message": prompt})

        answer = response.get("answer", "抱歉，无法获取回答。")
        sources = response.get("sources", [])
        intent = response.get("intent", "knowledge_qa")

        st.markdown(answer)

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
        "content": answer,
        "intent": intent,
        "sources": sources,
    })
