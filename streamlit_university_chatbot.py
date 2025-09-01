import streamlit as st
import os
from typing import List, Dict

# ---------------------------
# Fixed UniLife Q&A Chatbot (Streamlit) - Gemini-ready
# - Prefers Streamlit secrets for GEMINI_API_KEY
# - Robust detection of several Gemini/GenAI SDK import patterns
# - Shows a non-secret key source indicator in the sidebar
# - Rule-based FAQ fallback
# ---------------------------

CUSTOM_KB: List[Dict[str, str]] = [
    {"q": "Where is the library?", "a": "The main university library is located in Building A, Level 2. Opening hours: Mon-Fri 8:30am - 10:00pm, Sat 9:00am - 5:00pm. For borrowing & returns use your student card at the front desk or the self-checkout kiosks."},
    {"q": "How do I register for exams?", "a": "Exam registration is done via the student portal under 'Academic > Exam Registration'. Make sure you've paid all necessary fees and completed course enrollment before the registration deadline. Contact the Exams Office if you face issues."},
    {"q": "How can I reset my student portal password?", "a": "Reset your password at the portal's 'Forgot Password' link. If that fails, submit a help ticket to IT Support with your student ID and a photo ID for verification."},
    {"q": "Where can I apply for scholarships?", "a": "Visit the Scholarships page under Student Services for current openings. Some scholarships require faculty nomination — check eligibility carefully and prepare transcripts and recommendation letters."},
    {"q": "What are the library's rules for group study rooms?", "a": "Group study rooms can be booked online via the library booking system for up to 2 hours at a time. Keep noise to a minimum and leave the room tidy. No food or drinks allowed in certain rooms — check room details when booking."},
]

FAQ_KEYWORDS = {
    "library": CUSTOM_KB[0]["a"],
    "exam": CUSTOM_KB[1]["a"],
    "register": CUSTOM_KB[1]["a"],
    "password": CUSTOM_KB[2]["a"],
    "scholarship": CUSTOM_KB[3]["a"],
    "group study": CUSTOM_KB[4]["a"],
    "study room": CUSTOM_KB[4]["a"],
}


def get_rule_based_answer(question: str) -> str:
    q = question.lower()
    for entry in CUSTOM_KB:
        if entry["q"].lower() in q or any(word in q for word in entry["q"].lower().split() if len(word) > 3):
            return entry["a"]
    for k, a in FAQ_KEYWORDS.items():
        if k in q:
            return a
    return "I don't have a precise answer in my FAQ. Try rephrasing the question or provide a Gemini API key in the sidebar to enable AI-powered answers."


def normalize_key(k: str | None) -> str | None:
    if not k:
        return None
    k = k.strip()
    if (k.startswith('"') and k.endswith('"')) or (k.startswith("'") and k.endswith("'")):
        return k[1:-1]
    return k


def detect_gemini_key(sidebar_key: str | None) -> (str | None, str | None):
    """Return (effective_key, source_label) -- source_label is one of st.secrets, sidebar, env, none."""
    effective = None
    source = None
    try:
        if hasattr(st, "secrets") and st.secrets:
            if "GEMINI_API_KEY" in st.secrets:
                effective = st.secrets.get("GEMINI_API_KEY")
                source = "st.secrets"
            elif "GOOGLE_API_KEY" in st.secrets:
                effective = st.secrets.get("GOOGLE_API_KEY")
                source = "st.secrets"
    except Exception:
        pass
    if not effective and sidebar_key:
        effective = sidebar_key
        source = "sidebar"
    if not effective:
        effective = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if effective:
            source = "env"
    effective = normalize_key(effective)
    return effective, source


def call_gemini(question: str, api_key: str | None, model: str = "gemini-2.5-flash", temperature: float = 0.2) -> str:
    key, source = detect_gemini_key(api_key)
    # Provide a non-sensitive indicator in the sidebar
    try:
        if source:
            st.sidebar.info(f"Gemini key source: {source}")
        else:
            st.sidebar.warning("Gemini key not found (set st.secrets or env or paste in sidebar)")
    except Exception:
        pass

    if not key:
        return "(Gemini not configured) No Gemini API key found. Provide GEMINI_API_KEY/GOOGLE_API_KEY as a Streamlit secret, env var, or paste key in the sidebar."

    # Try multiple SDK import patterns. Collect errors and return a clear fallback if none succeed.
    last_errors: list[str] = []

    # Pattern A: from google import genai
    try:
        from google import genai  # type: ignore
        try:
            client = None
            try:
                client = genai.Client()
            except Exception:
                try:
                    genai.configure(api_key=key)
                    client = genai.Client()
                except Exception as e:
                    client = None
                    last_errors.append(f"genai client init failed: {e}")
            if client is not None:
                resp = client.models.generate_content(model=model, contents=question)
                if hasattr(resp, "text"):
                    return str(resp.text)
                if hasattr(resp, "output"):
                    out = getattr(resp, "output")
                    candidates = getattr(out, "candidates", None)
                    if candidates:
                        texts = [getattr(c, "text", str(c)) for c in candidates]
                        return "\n".join(texts)
                return str(resp)
        except Exception as e:
            last_errors.append(f"genai.Client pattern runtime error: {e}")
    except Exception as e:
        last_errors.append(f"genai import not available: {e}")

    # Pattern B: import google.generativeai as genai
    try:
        import google.generativeai as genai2  # type: ignore
        try:
            if hasattr(genai2, "configure"):
                try:
                    genai2.configure(api_key=key)
                except Exception as e:
                    # keep going; configure may not be required or may raise
                    last_errors.append(f"genai2.configure warning: {e}")
            if hasattr(genai2, "generate_text"):
                resp = genai2.generate_text(model=model, input=question, temperature=temperature)
                if isinstance(resp, str):
                    return resp
                if hasattr(resp, "text"):
                    return str(resp.text)
                if isinstance(resp, dict):
                    return str(resp.get("text") or resp)
                return str(resp)
            if hasattr(genai2, "chat") and hasattr(genai2.chat, "create"):
                chat_resp = genai2.chat.create(model=model, messages=[{"role": "user", "content": question}], temperature=temperature)
                if hasattr(chat_resp, "content"):
                    return str(chat_resp.content)
                if isinstance(chat_resp, dict):
                    return str(chat_resp.get("output") or chat_resp)
                return str(chat_resp)
        except Exception as e:
            last_errors.append(f"google.generativeai runtime error: {e}")
    except Exception as e:
        last_errors.append(f"google.generativeai import not available: {e}")


    # Pattern C: google.ai.generativelanguage (TextServiceClient)
    # try:
    #     from google.ai import generativelanguage as gal  # type: ignore
    #     try:
    #         client = gal.TextServiceClient()
    #         if hasattr(client, "generate_text"):
    #             # This is a best-effort call; the exact request object may vary by version
    #             resp = client.generate_text(model=model, prompt=question)
    #             return str(resp)
    #     except Exception as e:
    #         return f"(Gemini request failed - google.ai.generativelanguage pattern) {e}"
    # except Exception:
    #     pass

    error_details = "; ".join(last_errors) if last_errors else "no details available"

    return (
        "(Gemini client not found) Could not find a supported Gemini/GenAI Python package "
        "or all client attempts failed. To enable Gemini in this app: `pip install google-generativeai` "
        "and set GEMINI_API_KEY or GOOGLE_API_KEY (Streamlit secrets or env). "
        f"Details: {error_details}"
    )

def main():
    st.set_page_config(page_title="UniLife Q&A Bot (Gemini)", layout="centered")
    st.title("🎓 UniLife — Student Q&A Chatbot (Gemini)")

    st.markdown(
        "Ask questions about university life (library, exams, scholarships, IT help, etc.).\n\n"
        "This app supports a rule-based FAQ and an optional Google Gemini integration (if you provide an API key and have the SDK installed)."
    )

    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox("Mode", ["Gemini (Google)", "Rule-based (FAQ)"], index=1)
    gemini_api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password")
    gemini_model = st.sidebar.text_input("Gemini model", value="gemini-2.5-flash")
    temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)

    st.sidebar.markdown("**Custom knowledge - included in the model prompt (used by AI):**")
    for e in CUSTOM_KB:
        st.sidebar.write(f"- {e['q']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Tips: If you don't have an API key or the SDK installed, choose Rule-based (FAQ). To install the SDK: `pip install google-generativeai`.")

    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    st.text_input("Type your question here", key="user_input")

    def handle_ask(mode_value, api_key_value, model_value, temperature_value):
        user_q = st.session_state.get('user_input', "").strip()
        if not user_q:
            return
        if mode_value == "Gemini (Google)":
            answer = call_gemini(user_q, api_key=api_key_value or None, model=model_value or "gemini-2.5-flash", temperature=temperature_value)
        else:
            answer = get_rule_based_answer(user_q)
        st.session_state['history'].append({"user": user_q, "bot": answer})
        st.session_state['user_input'] = ""

    st.button("Ask", on_click=handle_ask, args=(mode, gemini_api_key, gemini_model, temp))

    st.subheader("Conversation")
    if not st.session_state['history']:
        st.info("No messages yet. Ask something to start the chat.")
    else:
        for i, turn in enumerate(st.session_state['history'][::-1]):
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Bot:** {turn['bot']}")
            st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear conversation"):
            st.session_state['history'] = []
            st.experimental_rerun()
    with col2:
        if st.session_state['history']:
            convo_text = "\n\n".join([f"You: {h['user']}\nBot: {h['bot']}" for h in st.session_state['history']])
            st.download_button("Download conversation (txt)", convo_text, file_name="unilife_conversation.txt")
        else:
            st.write("")

    st.markdown("---")
    st.markdown("**Developer notes:** For Streamlit Cloud: add GEMINI_API_KEY to app Secrets. Avoid pasting API keys on public/shared machines.")


if __name__ == '__main__':
    main()


