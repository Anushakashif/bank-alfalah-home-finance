# app.py — Streamlit frontend (imports RAG logic from rag.py)

import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
from rag import build_vector_store, ask

# ── Load environment variables ─────────────────────────────────────────────────
load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Alfalah Home Finance Assistant",
    page_icon="🏠",
    layout="centered"
)

# ── Groq Client ────────────────────────────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Build Vector Store once and cache it ──────────────────────────────────────
@st.cache_resource
def load_vector_store():
    return build_vector_store(client)

vector_store = load_vector_store()

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None  # stores question from suggestion buttons

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🏠 Bank Alfalah — Home Finance Assistant")
st.caption("Powered by RAG · Ask me anything about home loan rates, eligibility, documents, and more.")
st.divider()

# ── Suggested Questions (shown only at start) ──────────────────────────────────
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    suggestions = [
        "What is the markup rate for salaried?",
        "What is the minimum income required?",
        "What documents do I need?",
        "Can I add my spouse as co-borrower?",
        "Is there a special offer for women?",
        "How do I apply for home finance?",
    ]
    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, use_container_width=True):
            # Save question as pending — do NOT rerun yet
            # We process it below in the same rerun
            st.session_state.pending_prompt = s
    st.divider()

# ── Render Chat History ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Policy sections used"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

# ── Resolve prompt — either from chat input or suggestion button ───────────────
prompt = st.chat_input("Ask about home finance rates, eligibility, documents...")

# If a suggestion button was clicked, use that as the prompt
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None  # clear it so it doesn't repeat

# ── Process prompt ─────────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy..."):
            answer, sources = ask(
                query=prompt,
                vector_store=vector_store,
                chat_history=st.session_state.chat_history,
                client=client
            )
            st.markdown(answer)
            with st.expander("📄 Policy sections used"):
                for source in sources:
                    st.markdown(f"- {source}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.rerun()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Bank Alfalah")
    st.divider()
    st.markdown("📞 **Helpline:** 111-225-111")
    st.markdown("🌐 [Official Website](https://www.bankalfalah.com/personal-banking/loans/home-finance/)")
    st.markdown("📋 [Apply via RAPID Portal](https://www.bankalfalah.com)")
    st.divider()

    st.caption("Answers are based on Bank Alfalah's official Home Finance policy. For personalized advice, visit a branch.")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()