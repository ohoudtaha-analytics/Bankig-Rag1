# -*- coding: utf-8 -*-
"""
streamlit_app.py
------------------
Final step of the RAG pipeline: STREAMLIT UI.

A simple chat-style assistant for bank underwriting/credit-risk staff to
ask natural-language questions about the Central Bank of Egypt's official
"Book 4 - Credit Granting Controls" regulation sections, and get answers
grounded in and cited to those sections (see 07_prompting.py for how
faithfulness/citation is enforced in the LLM prompt).

Deployment notes (see project instructions):
- Do NOT put your real OPENROUTER_API_KEY in this file or in a committed
  .env file.
- On Streamlit Cloud: App -> Manage app -> Secrets, and add:
    OPENROUTER_API_KEY = "your_openrouter_key_here"
    OPENROUTER_MODEL = "openai/gpt-4o-mini"
- Locally: set the OPENROUTER_API_KEY environment variable before running
  `streamlit run streamlit_app.py`.
"""

import importlib
import streamlit as st

# Modules 06_retrieve_context.py and 07_prompting.py start with a digit,
# which is not valid as a normal `import` statement target in Python, so
# they are loaded dynamically with importlib instead.
retrieve_module = importlib.import_module("06_retrieve_context")
prompting_module = importlib.import_module("07_prompting")

retrieve_context = retrieve_module.retrieve_context
generate_answer = prompting_module.generate_answer

# ---------------------------------------------------------------------------
# API key wiring: prefer Streamlit secrets when deployed, fall back to the
# environment variable already read inside 07_prompting.py for local dev.
# ---------------------------------------------------------------------------
try:
    if not prompting_module.OPENROUTER_API_KEY:
        prompting_module.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting_module.OPENROUTER_MODEL = st.secrets.get(
            "OPENROUTER_MODEL", prompting_module.OPENROUTER_MODEL
        )
except Exception:
    # st.secrets raises if no secrets.toml exists at all (e.g. pure local
    # run with only an environment variable set) - that's fine, ignore it.
    pass


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="مساعد سياسة الائتمان الذكي",
    page_icon="🏦",
    layout="centered",
)

st.title("🏦 مساعد سياسة الائتمان الذكي")
st.caption(
    "مساعد RAG يجيب على أسئلة موظفي الائتمان بالاستناد إلى النصوص الرسمية "
    "لضوابط منح الائتمان الصادرة عن البنك المركزي المصري (الباب الرابع - الفصل الأول)."
)

with st.sidebar:
    st.header("عن المشروع")
    st.write(
        "هذا المساعد يسترجع أقرب فقرات من ضوابط البنك المركزي المصري لمنح "
        "الائتمان، ثم يستخدمها كسياق لتوليد إجابة موثقة بمصدرها."
    )
    top_k = st.slider("عدد المقاطع المسترجعة (top_k)", min_value=2, max_value=8, value=4)
    st.divider()
    st.caption(
        "⚠️ الإجابات مبنية على أقسام محددة من كتاب اللوائح فقط (وليس الكتاب كاملاً)، "
        "وهي أداة مساعدة وليست بديلاً عن الرجوع للنص الرسمي الكامل أو الشئون القانونية."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Render previous turns
# ---------------------------------------------------------------------------
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("📄 المصادر المسترجعة"):
                for src in turn["sources"]:
                    quality_flag = " ⚠️ (جودة نص منخفضة)" if src["text_quality"] == "low_quality" else ""
                    st.markdown(f"**{src['title']}**{quality_flag}")
                    st.text(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
question = st.chat_input("اسأل عن أي ضابط من ضوابط منح الائتمان...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في النصوص واسترجاع السياق..."):
            retrieved_chunks = retrieve_context(question, top_k=top_k)

        with st.spinner("جاري توليد الإجابة..."):
            answer = generate_answer(question, retrieved_chunks)

        st.write(answer)

        with st.expander("📄 المصادر المسترجعة"):
            for chunk in retrieved_chunks:
                quality_flag = " ⚠️ (جودة نص منخفضة)" if chunk["text_quality"] == "low_quality" else ""
                st.markdown(f"**{chunk['title']}**{quality_flag}")
                st.text(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": retrieved_chunks,
    })
