import os
import io
import math
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# -----------------------------
# Indstillinger (kan tweakes)
# -----------------------------
MIN_DF = 5
MAX_FEATURES = 40000
NGRAM_RANGE = (1, 2)
K_RANGE = range(8, 26)
TOP_TERMS_PER_CAT = 8
TOP_QUOTES_PER_CAT = 3
PRIMARY_THRESHOLD = 0.15
SECOND_THRESHOLD = 0.12
THIRD_THRESHOLD  = 0.10

DANISH_SW = {
    "og","i","jeg","det","at","en","den","til","er","som","på","de","med","der",
    "har","for","af","ikke","et","men","vi","kan","om","så","ud","over","sig",
    "fra","bliver","eller","hvad","hvordan","hvorfor","når","man","noget",
    "meget","mere","mest","alle","alt","andre","også","kun","bare","ind","op",
    "ned","hos","hen","derfor","fordi","kunne","skulle","ville","bl.a.","fx","f.eks."
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | DANISH_SW

# -----------------------------
# Adgangskontrol (password via Secrets)
# -----------------------------
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

def check_auth():
    if APP_PASSWORD is None:
        st.warning("⚠️ Ingen adgangskode sat (APP_PASSWORD). App'en er åben for alle med link.")
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.title("🔒 Adgang påkrævet")
    pwd = st.text_input("Indtast adgangskode", type="password")
    if st.button("Log ind"):
        if pwd == APP_PASSWORD:
            st.session_state["auth_ok"] = True
            st.session_state.setdefault("audit", [])
            st.session_state["audit"].append({"ts": datetime.utcnow().isoformat(), "action": "login_success"})
            st.rerun()
        else:
            st.session_state.setdefault("audit", [])
            st.session_state["audit"].append({"ts": datetime.utcnow().isoformat(), "action": "login_fail"})
            st.error("Forkert adgangskode")
    return False

if not check_auth():
    st.stop()

# -----------------------------
# Logging-hjælpere
# -----------------------------
def log_event(action, **kwargs):
    st.session_state.setdefault("audit", [])
    row = {"ts": datetime.utcnow().isoformat(), "action": action}
    row.update(kwargs)
    st.session_state["audit"].append(row)

def download_audit_widget():
    if not st.session_state.get("audit"):
        st.info("Ingen log endnu.")
        return
    df_log = pd.DataFrame(st.session_state["audit"])
    csv = df_log.to_csv(index=False).encode("utf-8")
    st.download_button("Download session-log (CSV)", data=csv,
                       file_name="session_audit_log.csv", mime="text/csv")

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Åbne svar → Kategorier (secure)", layout="wide")
st.title("Åbne besvarelser → Auto-kategorisering")
st.caption("Version med adgangskode og session-log.")

st.markdown(
    "Upload et Excel-ark med en kolonne med åbne svar (fx \"Svar\"). "
    "App'en finder kategorier automatisk, tildeler op til 3 pr. svar og giver et Analyse-ark."
)

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])

use_llm = st.toggle("Brug LLM til pænere kategorinavne (valgfrit)", value=False)
llm_model = st.text_input("LLM model-navn (fx 'gpt-4o-mini')", value="gpt-4o-mini", disabled=not use_llm)
llm_temperature = st.slider("LLM temperatur", 0.0, 1.0, 0.2, 0.05, disabled=not use_llm)

log_event("llm_toggle", enabled=bool(use_llm), model=llm_model if use_llm else None,
          temperature=float(llm_temperature) if use_llm else None)

# -----------------------------
# Hjælpefunktioner
# -----------------------------
def read_excel_first_text_column(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    if df.shape[1] == 0:
        raise ValueError("Excel-arket er tomt.")
    candidates = [c for c in df.columns if str(c).strip().lower() in {"svar","tekst","text","response","kommentar"}]
    col = candidates[0] if candidates else df.columns[0]
    series = df[col].astype(str).fillna("").map(lambda s: s.strip())
    series = series[series.str.len() > 0]
    return pd.DataFrame({"Svar": series})

def top_terms_for_center(center_vec, feature_names, topn=8):
    idx = np.argsort(center_vec)[::-1][:topn]
    return [feature_names[i] for i in idx if center_vec[i] > 0]

def auto_label_from_terms(terms):
    base = ", ".join([t for t in terms[:3]])
    titled = " / ".join([w.strip().capitalize() for w in base.split(",") if w.strip()])
    return titled if titled else "Andet"

def llm_label_from_terms(terms, model="gpt-4o-mini", temperature=0.2):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            "Du er en dansktalende analytiker. Givet disse top-termer for en kategori, "
            "foreslå et kort, præcist kategorinavn (2-5 ord) uden citationstegn.\n\n"
            f"Top-termer: {', '.join(terms)}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=16,
        )
        name = resp.choices[0].message.content.strip()
        name = name.replace('"', '').replace("'", "")
        return name
    except Exception:
        return None

def build_analysis_sheet(df_data, categories_info):
    total = len(df_data)
    counts = Counter(df_data["Kategori_1"].fillna("Andet"))
    rows = []
    for cat, info in categories_info.items():
        cnt = counts.get(cat, 0)
        pct = (cnt / total * 100.0) if total else 0.0
        terms = ", ".join(info.get("top_terms", [])[:TOP_TERMS_PER_CAT])
        mask = df_data["Kategori_1"] == cat
        examples = (
            df_data.loc[mask]
                  .sort_values("Score_1", ascending=False)["Svar"]
                  .head(TOP_QUOTES_PER_CAT)
                  .tolist()
        )
        while len(examples) < TOP_QUOTES_PER_CAT:
            examples.append("")
        rows.append({
            "Kategori": cat,
            "Antal": cnt,
            "Andel %": round(pct, 1),
            "Top-termer": terms,
            "Citater 1": examples[0],
            "Citater 2": examples[1],
            "Citater 3": examples[2],
        })
    rows = sorted(rows, key=lambda r: r["Antal"], reverse=True)
    return pd.DataFrame(rows)

# -----------------------------
# Hovedlogik
# -----------------------------
if uploaded:
    log_event("file_upload", filename=getattr(uploaded, "name", "unknown"))

    with st.spinner("Indlæser og analyserer..."):
        df = read_excel_first_text_column(uploaded)
        texts = df["Svar"].tolist()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOP_WORDS),
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            max_features=MAX_FEATURES
        )
        X = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # vælg automatisk k via silhouette
        best_k = None
        best_score = -1
        best_km = None
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labs = km.fit_predict(X)
            try:
                score = silhouette_score(X, labs, metric="cosine",
                                         sample_size=min(10000, X.shape[0]))
            except Exception:
                score = -1
            if score > best_score:
                best_k, best_score, best_km = k, score, km

        km = best_km
        centers = km.cluster_centers_
        centers_norm = normalize(centers, norm="l2", axis=1)
        X_norm = normalize(X, norm="l2", axis=1)

        # opbyg kategorier
        categories_info = {}
        id2label = {}
        for cid in range(best_k):
            terms = top_terms_for_center(centers[cid], feature_names,
                                         topn=TOP_TERMS_PER_CAT)
            label = None
            if use_llm:
                label = llm_label_from_terms(terms, model=llm_model,
                                             temperature=llm_temperature)
            if not label:
                label = auto_label_from_terms(terms)
            base = label
            suffix = 2
            while label in categories_info:
                label = f"{base} ({suffix})"
                suffix += 1
            categories_info[label] = {"id": cid, "top_terms": terms}
            id2label[cid] = label

        sims = X_norm @ centers_norm.T

        def top3_for_row(row_idx):
            row = sims[row_idx]
            idx_sorted = np.argsort(-row)[:3]
            labels3 = [id2label.get(i, "Andet") for i in idx_sorted]
            scores3 = [float(row[i]) for i in idx_sorted]
            out_labels, out_scores = [], []
            thresholds = [PRIMARY_THRESHOLD, SECOND_THRESHOLD, THIRD_THRESHOLD]
            for lab, sc, th in zip(labels3, scores3, thresholds):
                if sc >= th:
                    out_labels.append(lab)
                    out_scores.append(sc)
            if not out_labels:
                out_labels = ["Andet"]
                out_scores = [0.0]
            while len(out_labels) < 3:
                out_labels.append("")
                out_scores.append(np.nan)
            return out_labels, out_scores

        cats_1, cats_2, cats_3 = [], [], []
        sc_1, sc_2, sc_3 = [], [], []
        for i in range(sims.shape[0]):
            labs, scs = top3_for_row(i)
            cats_1.append(labs[0]); cats_2.append(labs[1]); cats_3.append(labs[2])
            sc_1.append(scs[0]);   sc_2.append(scs[1]);   sc_3.append(scs[2])

        df_out = pd.DataFrame({
            "Svar": texts,
            "Kategori_1": cats_1,
            "Kategori_2": cats_2,
            "Kategori_3": cats_3,
            "Score_1": sc_1,
            "Score_2": sc_2,
            "Score_3": sc_3,
        })

        analysis_df = build_analysis_sheet(df_out, categories_info)

        st.subheader("Eksempel på output (første 20 rækker)")
        st.dataframe(df_out.head(20), use_container_width=True)

        st.subheader("Analyse pr. kategori")
        st.dataframe(analysis_df, use_container_width=True)

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Data", index=False)
            meta = pd.DataFrame([{
                "Dato": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "Antal svar": len(df_out),
                "Antal kategorier (auto)": best_k,
                "Silhouette (cosine)": round(float(best_score), 3) if isinstance(best_score, (int,float,np.floating)) and not math.isnan(best_score) else ""
            }])
            meta.to_excel(writer, sheet_name="Analyse", index=False, startrow=0)
            analysis_df.to_excel(writer, sheet_name="Analyse", index=False, startrow=3)
        buffer.seek(0)

        st.download_button(
            "Download Excel med kategorier og analyse",
            data=buffer,
            file_name="aabne_svar_kategoriseret.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click=lambda: log_event("download_clicked", rows=len(df_out), k=best_k)
        )

st.divider()
st.subheader("Session-log")
download_audit_widget()
