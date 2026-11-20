import streamlit as st
import os
import time
from datetime import date
from streamlit.errors import StreamlitSecretNotFoundError
from openai import OpenAI
from utils import get_image_description, resolve_api_key

DEFAULT_PROMPT =  """
Analyze the image and generate a classified advertisement in simple, clear, and natural language.

Include the following sections:

1. Title: Create a short, informative, and attractive listing title that identifies the main object.
2. Description: Describe the object, its visible features, condition, and possible use. Focus on details relevant to potential buyers.
3. Specifications: List relevant attributes depending on the object type, such as area, number of rooms, condition, brand, model, color, or year of manufacture.

Use only information that can be reasonably determined from the image.
Do not invent missing details.

Generate the entire advertisement in Polish.
""".strip()

NO_TITLE_PROMPT = """
Do not use any user-provided title. Generate the listing title yourself based only on the image.
""".strip()

WITH_TITLE_PROMPT = """
Use the following user-provided title as the main idea for the advertisement:
"{title}"

The title must remain the central product or property theme. You may correct grammar or make it
more natural, but do not change its meaning or invent facts that are not visible or implied by
the title and image.
""".strip()

FALLBACK_MODELS = ["gpt-6-luna", "gpt-6.1-terra", "gpt-6.2-sol"]
DEFAULT_MODEL = "gpt-6-luna"

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default

DISABLE_MANUAL_API_KEY_INPUT = (
    str(get_secret("DISABLE_MANUAL_API_KEY_INPUT", "")).lower() in {"1", "true", "yes"}
)

def get_int_config(name, default):
    raw_value = get_secret(name, os.environ.get(name, default))
    try:
        value = int(raw_value)
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default

MAX_REQUESTS_PER_MINUTE = get_int_config("MAX_REQUESTS_PER_MINUTE", 5)
MAX_REQUESTS_PER_DAY = get_int_config("MAX_REQUESTS_PER_DAY", 100)

def fetch_available_gpt_models(client):
    models = client.models.list()
    available_model_ids = {model.id for model in models.data}
    allowed_models = [model for model in FALLBACK_MODELS if model in available_model_ids]
    return allowed_models

# Streamlit app layout
st.set_page_config(
    page_title="Generator ogłoszeń AI",
    page_icon=":camera:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(99, 102, 241, .14), transparent 28rem),
            radial-gradient(circle at 100% 10%, rgba(14, 165, 233, .12), transparent 24rem),
            #f8fafc;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 2.25rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, .16);
        border-radius: 28px;
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 52%, #0284c7 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(49, 46, 129, .20);
    }
    .hero .eyebrow {
        margin: 0 0 .45rem;
        color: #c7d2fe;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.08;
    }
    .hero p {
        max-width: 680px;
        margin: 1rem 0 0;
        color: #e0e7ff;
        font-size: 1.05rem;
    }
    .info-card {
        min-height: 100px;
        padding: 1.1rem 1.25rem;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        background: rgba(255, 255, 255, .82);
        box-shadow: 0 8px 24px rgba(15, 23, 42, .06);
    }
    .info-card strong {
        display: block;
        margin-bottom: .35rem;
        color: #1e293b;
        font-size: 1rem;
    }
    .info-card span {
        color: #64748b;
        font-size: .9rem;
    }
    [data-testid="stFileUploader"] {
        padding: .35rem;
        border: 2px dashed #c7d2fe;
        border-radius: 20px;
        background: rgba(255, 255, 255, .72);
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: #eef2ff;
    }
    .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 0;
        font-weight: 700;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(79, 70, 229, .22);
    }
    </style>
    <div class="hero">
        <p class="eyebrow">AI powered marketplace studio</p>
        <h1>Generator ogłoszeń ze zdjęcia</h1>
        <p>Dodaj zdjęcie, a przygotujemy przejrzyste i gotowe do publikacji ogłoszenie po polsku.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

info_left, info_middle, info_right = st.columns(3)
with info_left:
    st.markdown(
        '<div class="info-card"><strong>01 · Dodaj zdjęcie</strong>'
        '<span>Obsługujemy JPG, PNG i WEBP.</span></div>',
        unsafe_allow_html=True,
    )
with info_middle:
    st.markdown(
        '<div class="info-card"><strong>02 · Dodaj kontekst</strong>'
        '<span>Opcjonalny tytuł pomoże dopasować treść.</span></div>',
        unsafe_allow_html=True,
    )
with info_right:
    st.markdown(
        '<div class="info-card"><strong>03 · Pobierz wynik</strong>'
        '<span>Otrzymasz tekst gotowy do publikacji.</span></div>',
        unsafe_allow_html=True,
    )

st.write("")
st.caption(
    f"Rate limits: {MAX_REQUESTS_PER_MINUTE} request(s)/minute and {MAX_REQUESTS_PER_DAY} request(s)/day per user session."
)

api_key = resolve_api_key()
if not api_key:
    api_key = get_secret("OPENAI_API_KEY", "")

if not DISABLE_MANUAL_API_KEY_INPUT:
    manual_api_key = st.text_input(
        "OpenAI API key (optional if OPENAI_API_KEY is set)",
        type="password",
        help="If empty, the app will use OPENAI_API_KEY from environment variables or Streamlit secrets.",
    )
    if manual_api_key:
        api_key = manual_api_key

if api_key:
    client = OpenAI(api_key=api_key)
    try:
        available_models = fetch_available_gpt_models(client)
    except Exception as e:
        available_models = FALLBACK_MODELS
        st.warning(f"Could not fetch models from API. Using fallback list. Details: {e}")

    if not available_models:
        available_models = FALLBACK_MODELS
        st.warning("No GPT models were returned by the API. Using fallback model list.")

    default_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else available_models[0]
    default_index = available_models.index(default_model)
    model_choice = st.selectbox("Select model", available_models, index=default_index)
    custom_model = st.text_input(
        "Custom model ID (optional)",
        value="",
        help="If set, this value overrides the model selected above.",
    )

    prompt = DEFAULT_PROMPT

    st.subheader("Dodatkowe dane do ogłoszenia")
    listing_title = st.text_input(
        "Tytuł ogłoszenia (opcjonalny)",
        placeholder="np. Działka budowlana 1200 m²",
        help="Tytuł zostanie użyty jako dodatkowe wejście w wariancie porównawczym.",
    )
    compare_title_variants = st.checkbox(
        "Porównaj wynik bez tytułu i z tytułem",
        value=True,
        help="Generuje dwa warianty tego samego zdjęcia i modelu.",
    )

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption="Uploaded image", width="stretch")

            can_compare = compare_title_variants and bool(listing_title.strip())
            if compare_title_variants and not listing_title.strip():
                st.caption("Wpisz tytuł, aby włączyć oba warianty porównania.")

            if st.button("Wygeneruj ogłoszenie", type="primary"):
                if "request_timestamps" not in st.session_state:
                    st.session_state["request_timestamps"] = []
                if "daily_usage" not in st.session_state:
                    st.session_state["daily_usage"] = {"date": date.today().isoformat(), "count": 0}

                now = time.time()
                recent_timestamps = [
                    ts for ts in st.session_state["request_timestamps"] if now - ts < 60
                ]
                st.session_state["request_timestamps"] = recent_timestamps

                if len(recent_timestamps) >= MAX_REQUESTS_PER_MINUTE:
                    st.error("Rate limit reached: too many requests in the last minute. Please wait.")
                    st.stop()

                today = date.today().isoformat()
                if st.session_state["daily_usage"]["date"] != today:
                    st.session_state["daily_usage"] = {"date": today, "count": 0}

                request_count = 2 if can_compare else 1
                if st.session_state["daily_usage"]["count"] + request_count > MAX_REQUESTS_PER_DAY:
                    st.error("Daily limit reached for this session. Please try again tomorrow.")
                    st.stop()

                if len(recent_timestamps) + request_count > MAX_REQUESTS_PER_MINUTE:
                    st.error("Rate limit reached: two variants require two requests. Please wait.")
                    st.stop()

                st.session_state["request_timestamps"].extend([now] * request_count)
                st.session_state["daily_usage"]["count"] += request_count

                with st.spinner("Calling model..."):
                    final_prompt = prompt.strip() or DEFAULT_PROMPT
                    final_model = custom_model.strip() or model_choice

                    if can_compare:
                        no_title_prompt = f"{final_prompt}\n\n{NO_TITLE_PROMPT}"
                        with_title_prompt = (
                            f"{final_prompt}\n\n"
                            f"{WITH_TITLE_PROMPT.format(title=listing_title.strip())}"
                        )

                        no_title_result = get_image_description(
                            client, uploaded_file, no_title_prompt, final_model
                        )
                        uploaded_file.seek(0)
                        with_title_result = get_image_description(
                            client, uploaded_file, with_title_prompt, final_model
                        )

                        st.subheader("Porównanie wyników")
                        left, right = st.columns(2)
                        with left:
                            st.markdown("### Bez tytułu")
                            st.write(no_title_result)
                            st.download_button(
                                "Pobierz wariant bez tytułu",
                                no_title_result,
                                file_name="ogloszenie_bez_tytulu.txt",
                                mime="text/plain",
                                key="download_no_title",
                            )
                        with right:
                            st.markdown("### Z tytułem")
                            st.write(with_title_result)
                            st.download_button(
                                "Pobierz wariant z tytułem",
                                with_title_result,
                                file_name="ogloszenie_z_tytulem.txt",
                                mime="text/plain",
                                key="download_with_title",
                            )


                    else:
                        final_prompt_with_title = final_prompt
                        if listing_title.strip():
                            final_prompt_with_title = (
                                f"{final_prompt}\n\n"
                                f"{WITH_TITLE_PROMPT.format(title=listing_title.strip())}"
                            )
                        else:
                            final_prompt_with_title = f"{final_prompt}\n\n{NO_TITLE_PROMPT}"

                        description = get_image_description(
                            client, uploaded_file, final_prompt_with_title, final_model
                        )
                        st.subheader("Wygenerowane ogłoszenie")
                        st.write(description)
                        st.download_button(
                            "Pobierz ogłoszenie",
                            description,
                            file_name="ogloszenie.txt",
                            mime="text/plain",
                            key="download_single",
                        )
        except Exception as e:
            st.error(f"Error: {e}")
else:
    if DISABLE_MANUAL_API_KEY_INPUT:
        st.warning("No API key configured. Set OPENAI_API_KEY in environment variables or Streamlit secrets.")
    else:
        st.warning("Please provide an API key in the field above or set OPENAI_API_KEY locally.")
