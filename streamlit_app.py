"""
streamlit_app.py — Semantic Search Engine UI
"""

import requests
import streamlit as st

# ── Config ──────────────────────────────────────────────
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    /* Dark premium feel */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        font-family: 'Inter', sans-serif;
    }

    /* Result cards */
    .result-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(120, 80, 255, 0.25);
    }
    .result-title {
        color: #f0f0f0;
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.4;
        margin: 0 0 0.5rem 0;
    }
    .result-meta {
        color: #d1d5db;
        font-size: 0.9rem;
        margin: 0.3rem 0;
    }
    .result-score {
        color: #a78bfa;
        font-weight: 700;
        font-size: 0.88rem;
        margin-top: 0.6rem;
    }
    .result-categories {
        color: #9ca3af;
        font-size: 0.82rem;
        margin-top: 0.3rem;
    }

    /* Hero */
    .hero-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        background: rgba(52, 211, 153, 0.15);
        color: #34d399;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-badge-error {
        background: rgba(248, 113, 113, 0.15);
        color: #f87171;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────
@st.cache_data(ttl=600)
def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def api_search_cached(query: str, mode: str) -> dict | None:
    """Cached search to make UI switching instant."""
    try:
        r = requests.post(f"{API_URL}/search", json={"query": query, "mode": mode}, timeout=30)
        return r.json()
    except Exception:
        return None

# ═══════════════════════════════════════════════════════
#  MAIN UI
# ═══════════════════════════════════════════════════════

# Hero
st.markdown('<p class="hero-title">🔎 Semantic Search Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Powered by Qdrant · FastAPI</p>', unsafe_allow_html=True)

# Health check
health_status = api_health()
if health_status:
    mode_display = health_status.get("mode", "unknown").upper()
    st.markdown(
        f'<div style="text-align:center;margin-bottom:1.5rem;">'
        f'<span class="status-badge">● Online — {health_status["points"]:,} documents</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="text-align:center;margin-bottom:1.5rem;">'
        '<span class="status-badge status-badge-error">● Backend offline — is the API running?</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Search Input (Centered) ────────────────────────────
col_spacer_l, col_center, col_spacer_r = st.columns([1, 2, 1])

with col_center:
    # We use a form to prevent reload on every keystroke, but button search triggers it
    with st.form("search_form"):
        query_input = st.text_input("Query", placeholder="e.g. warm winter gloves for men", label_visibility="collapsed")
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

# ── Search Settings (Radio) ─────────────────────────────
# We put this AFTER search input or in a sidebar to avoid clutter
# Using session state to handle instant switching if results exist
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

mode_map = {"Keyword Search (BM25)": "sparse", "Semantic Search (Vector)": "dense"}
selected_mode_label = st.radio("Search Mode", list(mode_map.keys()), index=0, horizontal=True)
selected_mode = mode_map[selected_mode_label]

# Logic:
# 1. If 'submitted' (Search button pressed):
#    - Update session_state['last_query']
#    - Fetch BOTH dense and sparse results (in parallel ideally, but sequential is fast enough)
#    - Store them in session_state
# 2. If 'last_query' exists (user just switched radio button):
#    - Display results from session_state based on selected_mode
#    - No API call needed (instant!)

if submitted and query_input:
    st.session_state["last_query"] = query_input
    
    with st.spinner("Fetching results..."):
        # Fetch both!
        res_sparse = api_search_cached(query_input, "sparse")
        res_dense = api_search_cached(query_input, "dense")
        
        st.session_state["results_sparse"] = res_sparse
        st.session_state["results_dense"] = res_dense

# Display Results
if st.session_state.get("last_query"):
    query_text = st.session_state["last_query"]
    
    # Select usage data
    if selected_mode == "sparse":
        data = st.session_state.get("results_sparse")
    else:
        data = st.session_state.get("results_dense")
    
    if data and data.get("results"):
        st.markdown(f'<p style="text-align:center;color:#9ca3af;margin:1rem 0;">'
                    f'Found {len(data["results"])} results for "<b style="color:#c4b5fd">{query_text}</b>" ({selected_mode})</p>',
                    unsafe_allow_html=True)

        for i, r in enumerate(data["results"], 1):
            p = r["payload"]
            
            # Extract fields
            en = p.get("en", {})
            title = en.get("name", p.get("title", p.get("text", "—")))
            price = en.get("price", "")
            original_price = en.get("original_price", "")
            discount = en.get("discount_percentage", 0)
            brand = en.get("brand", p.get("brand", {}).get("name", ""))
            categories = en.get("categories", [])
            image_url = p.get("image", "")

            stats = p.get("stats", {})
            rating = stats.get("rating_score", "")
            comment_count = stats.get("comment_count", 0)

            # Build info parts
            price_html = f'<span style="color:#60a5fa;font-weight:700;">${price}</span>' if price else ""
            if original_price and discount and discount > 0:
                price_html += f' <span style="color:#6b7280;text-decoration:line-through;font-size:0.8rem;">${original_price}</span>'
                price_html += f' <span style="color:#34d399;font-size:0.8rem;">-{discount:.0f}%</span>'

            rating_html = f'⭐ {rating}' if rating else ""
            brand_html = f'🏷️ {brand}' if brand else ""
            comments_html = f'💬 {comment_count}' if comment_count else ""
            cat_str = " › ".join(categories) if categories else ""

            # Score formatting
            raw_score = r["score"]
            if selected_mode == "sparse":
                score_display = f"BM25 Score: {raw_score:.2f}"
            else:
                score_display = f"Similarity: {raw_score:.4f}"

            # Result Card
            card_html = (
                f'<div class="result-card">'
                f'<p class="result-title">{i}. {title}</p>'
                f'<p class="result-meta">{price_html}  {rating_html}  {brand_html}  {comments_html}</p>'
                f'<p class="result-categories">{cat_str}</p>'
                f'<p class="result-score">{score_display}</p>'
                f'</div>'
            )

            if image_url:
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    st.image(image_url, width=140)
                with col_info:
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.markdown(card_html, unsafe_allow_html=True)

    elif data:
        st.info("No results found.")
    else:
        # Avoid showing error if simple empty startup
        pass

# Footer
st.markdown(
    '<div style="text-align:center;color:#6b7280;margin-top:3rem;font-size:0.85rem;">'
    'Semantic Search Engine · Qdrant · FastAPI · Streamlit'
    '</div>',
    unsafe_allow_html=True,
)
