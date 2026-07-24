# -*- coding: utf-8 -*-
"""
07_prompting.py
-----------------
Step 7 of the RAG pipeline: PROMPTING.

Builds the final prompt from the retrieved context (06_retrieve_context.py)
and the user's question, then calls an LLM through OpenRouter to generate
an answer. The prompt explicitly instructs the model to:
  1. Answer ONLY from the provided context (no outside knowledge / no
     hallucinated rules), and say so clearly if the answer isn't in the
     context.
  2. Cite which section(s) the answer came from, so the underwriting
     employee reading the answer can verify it against the source policy.

API key handling follows the project instructions: the real key is never
hard-coded here. It's read from the OPENROUTER_API_KEY environment
variable (for local development) and, when deployed, streamlit_app.py
overrides these two module-level variables from st.secrets before calling
generate_answer().
"""

import os
import requests

# Read from environment variable for local development.
# NEVER hard-code the real key here or commit a .env file with it.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد ذكاء اصطناعي متخصص في الإجابة عن أسئلة موظفي الائتمان والتحليل الائتماني
بناءً فقط على النصوص الرسمية المقدمة لك من ضوابط ولوائح البنك المركزي المصري بشأن منح الائتمان.

القواعد الملزمة:
1) أجب فقط بناءً على السياق (Context) المقدم لك أدناه. لا تستخدم أي معلومة من معرفتك العامة.
2) إذا لم تجد إجابة واضحة للسؤال داخل السياق المقدم، صرّح بذلك بوضوح ولا تخترع إجابة.
3) اذكر دائماً في نهاية إجابتك اسم القسم/الأقسام (Source) التي استندت إليها في الإجابة.
4) أجب باللغة العربية بأسلوب مهني ومختصر ومباشر يناسب موظف ائتمان في بنك."""


def build_user_prompt(question: str, retrieved_chunks: list) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[مصدر {i} - {chunk['title']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    return f"""السياق (Context):
{context_text}

سؤال الموظف:
{question}

اكتب إجابة دقيقة ومختصرة بناءً على السياق أعلاه فقط، واذكر في النهاية أسماء الأقسام المصدر."""


def generate_answer(question: str, retrieved_chunks: list) -> str:
    """Call the LLM via OpenRouter with the retrieved context and return its answer."""
    if not OPENROUTER_API_KEY:
        return (
            "⚠️ لم يتم ضبط مفتاح OPENROUTER_API_KEY. "
            "أضِفه كمتغير بيئة محلياً أو في Streamlit secrets عند النشر."
        )

    user_prompt = build_user_prompt(question, retrieved_chunks)

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,  # low temperature: favor faithful, deterministic answers
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"⚠️ حدث خطأ أثناء الاتصال بالموديل: {e}"
    except (KeyError, IndexError):
        return "⚠️ لم يتم استلام رد صالح من الموديل."


if __name__ == "__main__":
    print(
        "This module is meant to be imported by streamlit_app.py.\n"
        "See streamlit_app.py for the full retrieve -> prompt -> generate flow."
    )
