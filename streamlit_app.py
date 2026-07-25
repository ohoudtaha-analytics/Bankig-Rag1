
    # -*- coding: utf-8 -*-
"""
streamlit_app.py
------------------
Final step of the RAG pipeline: STREAMLIT UI.

A branded assistant for bank underwriting/credit-risk staff to ask
natural-language questions about the Central Bank of Egypt's official
"Book 4 - Credit Granting Controls" regulation sections, and get answers
grounded in and cited to those sections (see 07_prompting.py for how
faithfulness/citation is enforced in the LLM prompt).

Deployment notes (see project instructions):
- Do NOT put your real OPENROUTER_API_KEY in this file or in a committed
  .env file.
- On Streamlit Cloud: App -> Manage app -> Secrets, and add:
    OPENROUTER_API_KEY = "your_openrouter_key_here"
    OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
- Locally: set the OPENROUTER_API_KEY environment variable before running
  `streamlit run streamlit_app.py`.
"""

import importlib
import streamlit as st

retrieve_module = importlib.import_module("06_retrieve_context")
prompting_module = importlib.import_module("07_prompting")

retrieve_context = retrieve_module.retrieve_context
generate_answer = prompting_module.generate_answer

try:
    if not prompting_module.OPENROUTER_API_KEY:
        prompting_module.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting_module.OPENROUTER_MODEL = st.secrets.get(
            "OPENROUTER_MODEL", prompting_module.OPENROUTER_MODEL
        )
except Exception:
    pass


st.set_page_config(
    page_title="مساعد سياسة الائتمان الذكي",
    page_icon="🏦",
    layout="centered",
)

INDEXED_SECTIONS = [
    "القسم الأول: ضوابط عامة لمنح الائتمان",
    "القسم الثاني: تمويل شركات التنمية العقارية",
    "القسم الثالث: الاستحواذ وتقييم الشهرة",
    "القسم السادس: العمليات الاستيرادية",
    "القسم الثامن: التمويل العقاري",
    "القسم التاسع: الشراء بالهامش",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Cairo', sans-serif;
        direction: rtl;
    }

    :root {
        --navy: #10233F;
        --navy-2: #18345C;
        --gold: #B08D3E;
        --paper: #F5F6F8;
        --ink: #1A1F29;
        --muted: #5B6472;
        --ok: #1F7A5C;
        --warn: #B4650C;
    }

    .stApp { background: var(--paper); }

    .hero {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
        border-radius: 18px;
        padding: 28px 30px 22px 30px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(16,35,63,0.18);
    }
    .hero-title {
        color: #ffffff;
        font-size: 30px;
        font-weight: 800;
        margin: 0 0 6px 0;
        display: flex; align-items: center; gap: 10px; justify-content: flex-end;
    }
    .hero-sub {
        color: #C9D3E0;
        font-size: 14.5px;
        line-height: 1.9;
        margin: 0;
    }
    .hero-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .badge {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.20);
        color: #EDEFF3;
        border-radius: 999px;
        padding: 5px 14px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .badge b { color: #F2D9A3; }

    .coverage-wrap { margin: 6px 0 18px 0; }
    .coverage-label {
        font-size: 12.5px; font-weight: 700; color: var(--muted);
        margin-bottom: 8px; text-align: right;
    }
    .pill-row { display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }
    .pill {
        background: #ffffff;
        border: 1px solid #E1E5EC;
        border-right: 3px solid var(--gold);
        color: var(--navy);
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 12px;
        font-weight: 600;
    }

    .suggested-label {
        font-size: 13px; font-weight: 700; color: var(--muted);
        margin: 4px 0 8px 0; text-align: right;
    }
    div[data-testid="stButton"] > button {
        border-radius: 10px;
        border: 1px solid #DDE2EA;
        background: #ffffff;
        color: var(--navy);
        font-family: 'Cairo', sans-serif;
        font-weight: 600;
        font-size: 13px;
        padding: 8px 10px;
        transition: all 0.15s ease;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--gold);
        color: var(--gold);
        background: #FFFBF2;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #E5E8EE;
        border-right: 4px solid var(--navy);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-title { font-weight: 700; color: var(--navy); font-size: 13.5px; margin-bottom: 4px; }
    .source-text { color: var(--ink); font-size: 13px; line-height: 1.8; }
    .quality-flag {
        display: inline-block; margin-right: 6px;
        background: #FDF1E4; color: var(--warn);
        border-radius: 6px; padding: 1px 8px; font-size: 11px; font-weight: 700;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-left: 1px solid #E5E8EE;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <p class="hero-title">🏦 مساعد سياسة الائتمان الذكي</p>
        <p class="hero-sub">
            يجيب على أسئلة موظفي الائتمان بالاستناد فقط إلى النصوص الرسمية لضوابط منح
            الائتمان الصادرة عن البنك المركزي المصري — كل إجابة موثقة بمصدرها.
        </p>
        <div class="hero-badges">
            <span class="badge"><b>{len(INDEXED_SECTIONS)}</b> أقسام رسمية مفهرسة</span>
            <span class="badge">مدعوم بتقنية <b>RAG</b></span>
            <span class="badge">إجابات <b>موثقة بالمصدر</b></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    st.markdown('<div class="coverage-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="coverage-label">📚 الأقسام المفهرسة حالياً</div>', unsafe_allow_html=True)
    pills_html = "".join(f'<span class="pill">{s}</span>' for s in INDEXED_SECTIONS)
    st.markdown(f'<div class="pill-row">{pills_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ لوحة التحكم")
    st.write(
        "يسترجع المساعد أقرب فقرات من ضوابط البنك المركزي المصري لمنح الائتمان، "
        "ثم يستخدمها كسياق لتوليد إجابة موثقة بمصدرها."
    )
    top_k = st.slider("عدد المقاطع المسترجعة (top_k)", min_value=2, max_value=8, value=4)
    st.divider()
    st.markdown("**📚 التغطية الحالية**")
    for s in INDEXED_SECTIONS:
        st.caption(f"• {s}")
    st.divider()
    st.caption(
        "⚠️ الإجابات مبنية على أقسام محددة من كتاب اللوائح فقط (وليس الكتاب كاملاً)، "
        "وهي أداة مساعدة وليست بديلاً عن الرجوع للنص الرسمي الكامل أو الشئون القانونية."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def render_sources(chunks):
    with st.expander(f"📄 المصادر المسترجعة ({len(chunks)})"):
        for chunk in chunks:
            quality_flag = (
                '<span class="quality-flag">⚠️ جودة نص منخفضة</span>'
                if chunk["text_quality"] == "low_quality" else ""
            )
            preview = chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else "")
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-title">{chunk['title']} {quality_flag}</div>
                    <div class="source-text">{preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def answer_question(question: str):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث في النصوص واسترجاع السياق..."):
            retrieved_chunks = retrieve_context(question, top_k=top_k)
        with st.spinner("جاري توليد الإجابة..."):
            answer = generate_answer(question, retrieved_chunks)
        st.write(answer)
        render_sources(retrieved_chunks)

    st.session_state.chat_history.append({
        "role": "assistant", "content": answer, "sources": retrieved_chunks,
    })


if not st.session_state.chat_history:
    st.markdown('<div class="suggested-label">💡 جرّبي أحد الأسئلة دي أو اكتبي سؤالك بالأسفل</div>', unsafe_allow_html=True)
    suggestions = [
        "ما الحد الأقصى لنسبة أقساط القروض الاستهلاكية إلى الدخل الشهري؟",
        "ما ضوابط تمويل عمليات الشراء بالهامش؟",
        "ما ضوابط منح الائتمان للاستحواذ على الشركات؟",
    ]
    cols = st.columns(len(suggestions))
    for col, s in zip(cols, suggestions):
        with col:
            if st.button(s, key=f"sugg_{s}", use_container_width=True):
                st.session_state.pending_question = s

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            render_sources(turn["sources"])

typed_question = st.chat_input("اسأل عن أي ضابط من ضوابط منح الائتمان...")

question_to_answer = st.session_state.pending_question or typed_question
if question_to_answer:
    st.session_state.pending_question = None
    answer_question(question_to_answer)
