"""
Tool:                 Invoice field extractor (Streamlit + pdfplumber + Groq, Colab prototype)
Purpose:              pdfplumber extracts invoice text; user reviews it; on "Analyze", Groq reads
                      the TEXT and returns labels (name, address, number, date, amount); export CSV.
Owner:                TBD
Change control:       TBD
AI assistance:        Drafted with Claude Enterprise, requires human review
Data classification:  Internal (test with SYNTHETIC invoices only)

Flow:  upload PDF -> pdfplumber text (shown) -> [Analyze] Groq reads text -> labels (shown) -> [Export CSV]
The Groq API key is entered in the front end each session and is never stored in this file.
"""

import io
import json

import pandas as pd
import streamlit as st

try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    from groq import Groq
except Exception:
    Groq = None


# ============================== theme (from reference image) ==============================
st.set_page_config(page_title="Invoice Reader", page_icon="🧾", layout="centered")

st.markdown(
    """
    <style>
      :root{
        --bg:#ededeb; --card:#ffffff; --ink:#1a1a1a; --muted:#6e6e73;
        --accent:#e8622c; --line:#e3e3e0;
      }
      .stApp { background: var(--bg); }
      .block-container { max-width: 820px; padding-top: 4.5rem; padding-bottom: 4rem; }
      /* hide Streamlit's top toolbar/header that overlaps content */
      header[data-testid="stHeader"]{ background: transparent; height: 0; }
      #MainMenu, footer{ visibility: hidden; }
      html, body, [class*="css"]{
        font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif;
        color:var(--ink);
      }
      /* force dark text everywhere so nothing turns white on white */
      .stApp, .stMarkdown, .stMarkdown p, label, .stCaption, p, span, div,
      .stDataFrame, [data-testid="stTable"], [data-testid="stTable"] * ,
      .stTextArea label, .stFileUploader label, .stTextInput label{
        color:var(--ink) !important;
      }
      .stCaption, [data-testid="stCaptionContainer"]{ color:var(--muted) !important; }
      /* text area + inputs: white field, dark text */
      textarea, .stTextArea textarea, div[data-baseweb="input"] input{
        color:var(--ink) !important; background:#faf9f7 !important;
        border-radius:12px !important; -webkit-text-fill-color:var(--ink) !important;
      }
      /* table cells */
      [data-testid="stTable"] td, [data-testid="stTable"] th{
        color:var(--ink) !important; background:#ffffff !important;
      }
      .display{ font-size:2.6rem; font-weight:800; letter-spacing:-0.03em; line-height:1.12;
                margin:0.5rem 0 0.25rem; padding-top:0.3rem; color:var(--ink) !important; }
      .sub{ color:var(--muted) !important; font-size:0.98rem; margin:0 0 1.75rem; }
      .card{ background:var(--card); border-radius:22px; padding:1.4rem 1.6rem;
             border:1px solid var(--line); margin-bottom:1.1rem; }
      .step{ font-size:0.78rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase;
             color:var(--accent) !important; margin:0 0 0.6rem; }
      .stButton>button{
        background:var(--accent) !important; color:#fff !important; border:none; border-radius:980px;
        padding:0.6rem 1.6rem; font-weight:600; font-size:0.95rem;
      }
      .stButton>button:hover{ background:#d1551f !important; color:#fff !important; }
      .stButton>button *{ color:#fff !important; }
      .stDownloadButton>button{
        background:var(--ink) !important; color:#fff !important; border:none; border-radius:980px;
        padding:0.6rem 1.6rem; font-weight:600;
      }
      .stDownloadButton>button *{ color:#fff !important; }
      [data-testid="stFileUploaderDropzone"]{
        background:#f5f5f3; border:1.5px dashed #c9c9c4; border-radius:16px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="display">Let\'s read<br>that invoice.</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">pdfplumber pulls the text, you review it, then Groq extracts the labels. '
    "Every value is a draft — verify it. Test invoices only.</p>",
    unsafe_allow_html=True,
)

missing = [n for n, m in [("pdfplumber", pdfplumber), ("groq", Groq)] if m is None]
if missing:
    st.error("Missing package(s): " + ", ".join(missing) +
             ".  Run:  !pip install -q streamlit pdfplumber groq pandas")
    st.stop()


# ============================== step 1: key ==============================
# Prefer a backend key stored in Streamlit Secrets (never in Git).
# If it isn't set, fall back to asking the user for one.
api_key = None
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = None

if not api_key:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="step">Step 1 · Groq API key</p>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Groq API key", type="password",
        placeholder="Paste your Groq key here (starts with gsk_...)",
        label_visibility="collapsed",
        help="Entered fresh each session. Never saved to the file or shared.",
    )
    st.markdown('</div>', unsafe_allow_html=True)
    if not api_key:
        st.info("Enter your Groq API key above to continue.")
        st.stop()

MODEL = "openai/gpt-oss-120b"   # confirmed available on this Groq account; fallback: openai/gpt-oss-20b


# ============================== helpers ==============================
EMPTY = {"customer_name": None, "customer_address": None,
         "invoice_or_order_number": None, "date": None,
         "total_amount": None, "currency": None}

SYSTEM = "You extract structured fields from invoice text and reply with JSON only."

PROMPT = """From the invoice text below, return a JSON object with EXACTLY these keys:
customer_name, customer_address, invoice_or_order_number, date, total_amount, currency.
Rules: use null if a field is absent; never guess. customer_name/address = the party billed (buyer),
not the seller. total_amount = final amount payable (grand total incl. tax) as a plain number.

INVOICE TEXT:
---
{text}
---"""


def extract_text(file_bytes):
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
        return txt.strip(), None
    except Exception as e:
        return "", f"could not read PDF ({e})"


def analyze(client, text):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": PROMPT.format(text=text[:20000])}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        out = dict(EMPTY); out.update({k: data.get(k) for k in EMPTY})
        return out, None
    except Exception as e:
        return dict(EMPTY), f"API error ({e})"


# ============================== step 2: upload + extract text ==============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="step">Step 2 · Upload & read PDF</p>', unsafe_allow_html=True)
f = st.file_uploader("Upload an invoice PDF", type=["pdf"], label_visibility="collapsed")

if f is not None:
    text, err = extract_text(f.getvalue())
    if err:
        st.error(err)
        st.session_state.pop("text", None)
    elif not text:
        st.warning("No text found — this looks like a scanned image, not a text PDF.")
        st.session_state.pop("text", None)
    else:
        st.session_state.text = text
        st.session_state.fname = f.name

if st.session_state.get("text"):
    st.caption("Extracted text (review before analyzing):")
    st.text_area("Extracted text", st.session_state.text, height=220,
                 label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)


# ============================== step 3: analyze ==============================
if st.session_state.get("text"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="step">Step 3 · Analyze with Groq</p>', unsafe_allow_html=True)
    if st.button("Analyze"):
        try:
            client = Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Could not start Groq client — check the key. ({e})")
            client = None
        if client:
            with st.spinner("Groq is reading the text..."):
                labels, aerr = analyze(client, st.session_state.text)
            st.session_state.labels = labels
            st.session_state.aerr = aerr

    if st.session_state.get("labels"):
        if st.session_state.get("aerr"):
            st.warning(st.session_state.aerr)
        st.caption("Extracted labels (draft — verify against the PDF):")
        st.table(pd.DataFrame([st.session_state.labels]).T.rename(columns={0: "value"}))
    st.markdown('</div>', unsafe_allow_html=True)


# ============================== step 4: export csv ==============================
if st.session_state.get("labels"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="step">Step 4 · Export</p>', unsafe_allow_html=True)
    row = dict(st.session_state.labels)
    row["source_file"] = st.session_state.get("fname", "")
    cols = ["source_file", "customer_name", "customer_address",
            "invoice_or_order_number", "date", "total_amount", "currency"]
    df = pd.DataFrame([row])[cols]
    st.download_button(
        "⬇  Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="invoice_extracted.csv",
        mime="text/csv",
    )
    st.markdown('</div>', unsafe_allow_html=True)
