import streamlit as st
import os
from typing import List, Dict

# Try to import Gemini client libraries (we'll handle failures gracefully at runtime)
# The Python ecosystem for Gemini has a few variants: `google.genai` / `google.generativeai` / `google` package with `genai` submodule.
# We will attempt imports at runtime inside the call_gemini function to avoid import-time crashes for users who don't have the packages installed.

# ---------------------------
# Simple University Q&A Chatbot (Streamlit) - Gemini edition
# - Supports Google Gemini via the GenAI SDK (if API key + package present)
# - Rule-based FAQ fallback
# - Stores conversation history in session_state
# - Fixes text input clearing issue by using a button callback
# ---------------------------

# ---------- Custom knowledge base (required: >=5 answers) ----------
CUSTOM_KB: List[Dict[str, str]] = [
    {
        "q": "Where is the library?",
        "a": "The main university library is located in Building A, Level 2. Opening hours: Mon-Fri 8:30am - 10:00pm, Sat 9:00am - 5:00pm. For borrowing & returns use your student card at the front desk or the self-checkout kiosks."
    },
    {
        "q": "How do I register for exams?",
        "a": "Exam registration is done via the student portal under 'Academic > Exam Registration'. Make sure you've paid all necessary fees and completed course enrollment before the registration deadline. Contact the Exams Office if you face issues."
    },
    {
        "q": "How can I reset my student portal password?",
        "a": "Reset your password at the portal's 'Forgot Password' link. If that fails, submit a help ticket to IT Support with your student ID and a photo ID for verification."
    },
    {
        "q": "Where can I apply for scholarships?",
        "a": "Visit the Scholarships page under Student Services for current openings. Some scholarships require faculty nomination — check eligibility carefully and prepare transcripts and recommendation letters."
    },
    {
        "q": "What are the library's rules for group study rooms?",
        "a": "Group study rooms can be booked online via the library booking system for up to 2 hours at a time. Keep noise to a minimum and leave the room tidy. No food or drinks allowed in certain rooms — check room details when booking."
    },
]

# ---------- Simple rule-based FAQ keywords ----------
FAQ_KEYWORDS = {
    "library": CUSTOM_KB[0]["a"],
    "exam": CUSTOM_KB[1]["a"],
    "register": CUSTOM_KB[1]["a"],
    "password": CUSTOM_KB[2]["a"],
    "scholarship": CUSTOM_KB[3]["a"],
    "group study": CUSTOM_KB[4]["a"],
    "study room": CUSTOM_KB[4]["a"],
}

# ---------- Helper: rule-based answer ----------

def get_rule_based_answer(question: str) -> str:
    q = question.lower()
    # Direct-ish match against full custom Q's
    for entry in CUSTOM_KB:
        if entry["q"].lower() in q or any(word in q for word in entry["q"].lower().split() if len(word) > 3):
            return entry["a"]

    # Keyword matching
    for k, a in FAQ_KEYWORDS.items():
        if k in q:
            return a

    return (
        "I don't have a precise answer in my FAQ. Try rephrasing the question or provide a Gemini API key in the sidebar to enable AI-powered answers."
    )


# ---------- Helper: call Google Gemini (Gen AI SDK) ----------

def call_gemini(question: str, api_key: str | None, model: str = "gemini-2.5-flash", temperature: float = 0.2) -> str:
    """
    Attempt to call Google Gemini using common Python SDK patterns. This function:
    - Looks for GEMINI_API_KEY or GOOGLE_API_KEY environment variables if api_key is None.
    - Attempts several import/call patterns so it works for a variety of installed gemini/genai packages.
    - Returns a helpful error string if the call fails (instead of crashing the app).

    Notes for users: Install the Google Gen AI SDK (one of):
      pip install google-generativeai
    or
      pip install google-genai

    And set your API key as an environment variable GEMINI_API_KEY or GOOGLE_API_KEY, or paste it into the sidebar.
    """
    effective_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not effective_key:
        return "(Gemini not configured) No Gemini API key found. Provide GEMINI_API_KEY/GOOGLE_API_KEY env var or paste key in the sidebar."

    # Try import pattern 1: `from google import genai` (common in official samples)
    try:
        from google import genai  # type: ignore

        # Configure client via environment or explicit configure if available
        try:
            # newer genai Client() pattern
            client = genai.Client()
        except Exception:
            # older configure pattern
            try:
                genai.configure(api_key=effective_key)
                client = genai.Client()
            except Exception:
                client = None

        if client is not None:
            # The generate_content API is common in samples
            try:
                resp = client.models.generate_content(model=model, contents=question)
                # Many samples expose `.text`
                if hasattr(resp, "text"):
                    return str(resp.text)
                # Some SDKs return objects/dicts — try common fields
                if hasattr(resp, "output"):
                    # join any text parts
                    out = getattr(resp, "output")
                    try:
                        # try to extract nested candidate text
                        candidates = getattr(out, "candidates", None)
                        if candidates:
                            texts = [c.text for c in candidates if hasattr(c, "text")]
                            return ".join(texts) if texts else str(resp)"
                    except Exception:
                        return str(resp)
                return str(resp)
            except Exception as e:
                return f"(Gemini request failed - genai.Client().models.generate_content) {e}"
    except Exception:
        # fall through and try alternative import patterns
        pass

    # Try import pattern 2: `import google.generativeai as genai`
    try:
        import google.generativeai as genai2  # type: ignore

        try:
            # configure if available
            if hasattr(genai2, "configure"):
                genai2.configure(api_key=effective_key)
        except Exception:
            pass

        # Try a couple of possible function names
        try:
            # Some older examples show `genai2.generate_text` or `genai2.chat.create`.
            if hasattr(genai2, "generate_text"):
                resp = genai2.generate_text(model=model, input=question, temperature=temperature)
                # try to return a textual representation
                if isinstance(resp, str):
                    return resp
                # try common attribute names
                if hasattr(resp, "text"):
                    return str(resp.text)
                if isinstance(resp, dict):
                    return str(resp.get("text") or resp)
                return str(resp)
            elif hasattr(genai2, "chat") and hasattr(genai2.chat, "create"):
                chat_resp = genai2.chat.create(model=model, messages=[{"role": "user", "content": question}], temperature=temperature)
                # try to extract text
                if hasattr(chat_resp, "content"):
                    return str(chat_resp.content)
                if isinstance(chat_resp, dict):
                    return str(chat_resp.get("output") or chat_resp)
                return str(chat_resp)
        except Exception as e:
            return f"(Gemini request failed - google.generativeai) {e}"
    except Exception:
        pass

    return (
        "(Gemini client not found) Could not find a supported Gemini/GenAI Python package. "
        "Install `google-generativeai` (or another official Gen AI SDK) and set GEMINI_API_KEY/GOOGLE_API_KEY."
    )


# ---------- Streamlit app UI (Gemini only) ----------

def main():
    st.set_page_config(page_title="UniLife Q&A Bot (Gemini)", layout="centered")
    st.title("🎓 UniLife — Student Q&A Chatbot (Gemini)")

    st.markdown(
        "Ask questions about university life (library, exams, scholarships, IT help, etc.)."
        "This app supports a rule-based FAQ and an optional Google Gemini integration (if you provide an API key and have the SDK installed)."
    )

    # Sidebar controls
    st.sidebar.header("Settings")
    mode = st.sidebar.selectbox("Mode", ["Gemini (Google)", "Rule-based (FAQ)"])

    gemini_api_key = st.sidebar.text_input("Gemini API Key (optional)", type="password")
    gemini_model = st.sidebar.text_input("Gemini model", value="gemini-2.5-flash")
    temp = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2)

    st.sidebar.markdown("**Custom knowledge - included in the model prompt (used by AI):**")
    for e in CUSTOM_KB:
        st.sidebar.write(f"- {e['q']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Tips: If you don't have an API key or the SDK installed, choose Rule-based (FAQ). To install the SDK: `pip install google-generativeai`. "
    )

    # Initialize session state for history and user_input if missing
    if 'history' not in st.session_state:
        st.session_state['history'] = []  # list of dicts: {user:..., bot:...}
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = ""

    # Input area (we will use a callback to handle submissions so we can clear the widget safely)
    st.text_input("Type your question here", key="user_input")

    # Define the callback handler that processes the question and updates history
    def handle_ask(mode_value, api_key_value, model_value, temperature_value):
        user_q = st.session_state.get('user_input', "").strip()
        if not user_q:
            # nothing to do
            return

        if mode_value == "Gemini (Google)":
            # Attempt to call Gemini (if no key provided, the function will return a helpful message)
            answer = call_gemini(user_q, api_key=api_key_value or None, model=model_value or "gemini-2.5-flash", temperature=temperature_value)
        else:
            answer = get_rule_based_answer(user_q)

        # Append to history
        st.session_state['history'].append({"user": user_q, "bot": answer})

        # Clear the input box by resetting the session_state value for the widget key
        # Doing this inside the callback is the correct pattern to avoid Streamlit's "cannot be modified after the widget is instantiated" error.
        st.session_state['user_input'] = ""

    # Place the Ask button that triggers the callback
    st.button("Ask", on_click=handle_ask, args=(mode, gemini_api_key, gemini_model, temp))

    # Show conversation history
    st.subheader("Conversation")
    if not st.session_state['history']:
        st.info("No messages yet. Ask something to start the chat.")
    else:
        for i, turn in enumerate(st.session_state['history'][::-1]):  # newest first
            st.markdown(f"**You:** {turn['user']}")
            st.markdown(f"**Bot:** {turn['bot']}")
            st.markdown("---")

    # Utilities: clear and download
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear conversation"):
            st.session_state['history'] = []
            st.experimental_rerun()
    with col2:
        if st.session_state['history']:
            convo_text = "\n".join([f"You: {h['user']}\nBot: {h['bot']}" for h in st.session_state['history']])
            st.download_button("Download conversation (txt)", convo_text, file_name="unilife_conversation.txt")

    st.markdown("---")
    st.markdown(
        "**Developer notes:** This app includes a small rule-based FAQ and an optional Google Gemini integration. For production: store API keys securely server-side, add rate-limiting, input validation, and content moderation before returning AI-generated responses."
    )


if __name__ == '__main__':
    main()

