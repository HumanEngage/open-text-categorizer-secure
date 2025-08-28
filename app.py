# app.py — Streamlit: Åbne besvarelser → ENTYDIGE kategorier
# - Password via Secrets: APP_PASSWORD="din-kode"
# - Valgfri LLM (OpenAI) til global, entydig navngivning: OPENAI_API_KEY="sk-..."
# - Flow: Auto-emner → LLM rydder op (1–3 ord, uden overlap) → vælg mål-antal (5–20)
#         → sammenlæg (bevar "Ved ikke" og "Andet") → multi-label tildeling → download

import os
import io
import re
import math
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances

# -------------------------------------------------
# Streamlit page config (skal kaldes tidligt)
# -------------------------------------------------
st.set_page_config(page_title="Åbne svar → Entydige kategorier (secure)", layout="wide")

# -------------------------------------------------
# Basale indstillinger (du kan tweake)
# -------------------------------------------------
MIN_DF = 5
MAX_FEATURES = 40000
NGRAM_RANGE = (1, 3)        # NU: også trigrammer (bedre fraser som "fri data")
K_RANGE = range(6, 16)      # lidt færre rå-klynger for mere robuste emner
TOP_TERMS_PER_CAT = 8
TOP_QUOTES_PER_CAT = 3

# Absolutte tærskler bruges stadig som bund – men vi vælger relativt i tildeling
PRIMARY_THRESHOLD = 0.18
SECOND_THRESHOLD  = 0.12
THIRD_THRESHOLD   = 0.10

# Stopord (dansk + engelsk)
DANISH_SW = {
    "og","i","jeg","det","at","en","den","til","er","som","på","de","med","der",
    "har","for","af","ikke","et","men","vi","kan","om","så","ud","over","sig",
    "fra","bliver","eller","hvad","hvordan","hvorfor","når","man","noget",
    "meget","mere","mest","alle","alt","andre","også","kun","bare","ind","op",
    "ned","hos","hen","derfor","fordi","kunne","skulle","ville","bl.a.","fx","f.eks."
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | DANISH_SW

# -------------------------------------------------
# Sikkerhed og session-log
# -------------------------------------------------
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

if not check_auth():
    st.stop()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Åbne besvarelser → Entydige kategorier")
st.caption("Auto-emner → LLM rydder op (1–3 ord, uden ord-overlap) → vælg mål-antal (5–20) → sammenlæg (bevar 'Ved ikke' og 'Andet') → multi-label → download.")
st.markdown(
    "Upload et Excel-ark med en kolonne med åbne svar (fx **Svar**). "
    "App'en finder rå-emner; LLM kan give entydige labels (1–3 ord, uden ord-overlap); "
    "du kan slå dem sammen til et ønsket antal kategorier."
)

uploaded = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
use_llm = st.toggle("Brug LLM til entydige, globale labels (kræver OPENAI_API_KEY)", value=False)
llm_model = st.text_input("LLM model-navn", value="gpt-4o-mini", disabled=not use_llm)
llm_temperature = st.slider("LLM temperatur", 0.0, 1.0, 0.2, 0.05, disabled=not use_llm)
log_event("llm_toggle", enabled=bool(use_llm), model=llm_model if use_llm else None,
          temperature=float(llm_temperature) if use_llm else None)

# -------------------------------------------------
# Hjælpefunktioner
# -------------------------------------------------
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

# Minimal fallback-navngivning, hvis LLM er slået fra/ikke virker
def auto_label_from_terms(terms):
    for t in terms:
        t2 = "".join(ch for ch in t if ch.isalpha())
        if len(t2) >= 3:
            return t2.capitalize()
    return "Andet"

# --- Label-normalisering og deduplikering (entydige labels) ---
def _strip_danish_suffix(word: str) -> str:
    w = word.lower().strip()
    w = re.sub(r"[^a-zA-ZæøåÆØÅ]+$", "", w)
    for suf in ["erne","enes","ende","tene","eren","er","en","et","e"]:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            w = w[:-len(suf)]
            break
    return w

def normalize_label(label: str) -> str:
    s = re.sub(r"[^A-Za-zÆØÅæøå\s]", " ", label).strip().lower()
    words = [w for w in re.split(r"\s+", s) if w]
    words = [_strip_danish_suffix(w) for w in words]
    words = [w for w in words if w][:3]  # maks 3 ord
    if not words:
        return "Andet"
    return " ".join(words).capitalize()

# --- Tekst-normalisering til tildeling (ord i teksten = hint) ---
def norm_token(w: str) -> str:
    w = w.lower()
    w = re.sub(r"[^a-zA-ZæøåÆØÅ]", "", w)
    for suf in ["erne","enes","ende","tene","eren","er","en","et","e"]:
        if w.endswith(suf) and len(w) > len(suf) + 1:
            w = w[:-len(suf)]
            break
    return w

def tokens_from_text(s: str):
    raw = re.findall(r"[A-Za-zÆØÅæøå]+", s.lower())
    return [norm_token(w) for w in raw if len(norm_token(w)) >= 2]

def contains_category_word(text: str, category_label: str) -> bool:
    tset = set(tokens_from_text(text))
    lab_toks = [norm_token(w) for w in re.findall(r"[A-Za-zÆØÅæøå]+", category_label)]
    return any(tok in tset for tok in lab_toks if tok)

# --- LLM navngivning globalt (1–3 ord, og ingen ord-overlap på tværs) ---
def _label_tokens(s: str):
    toks = re.findall(r"[A-Za-zÆØÅæøå]+", s.lower())
    return [t for t in toks if len(t) >= 2]

def _has_word_overlap(labels):
    seen = {}
    overlaps = []
    for i, lab in enumerate(labels):
        for w in set(_label_tokens(lab)):
            if w in seen:
                overlaps.append((seen[w], i, w))
            else:
                seen[w] = i
    return overlaps

def llm_name_categories_global(terms_list, counts_list, model="gpt-4o-mini", temperature=0.2):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return None

    lines = []
    for i, (terms, cnt) in enumerate(zip(terms_list, counts_list), 1):
        lines.append(f"{i}. ({cnt}) " + ", ".join(terms[:8]))
    spec = (
        "Du er dataanalytiker. Opgave: Navngiv hver kategori ENTydigt (1–3 danske ord).\n"
        "- Ingen samme ord må optræde i to forskellige kategorier.\n"
        "- Undgå tal/specialtegn. Brug korte, meningsfulde ord.\n"
        "- Returnér præcis én linje pr. kategori, samme antal linjer som input."
    )
    prompt = spec + "\n\nKategorier med top-termer og antal:\n" + "\n".join(lines)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
        txt = resp.choices[0].message.content.strip()
        labels = [l.strip().strip("-•0123456789. ").strip() for l in txt.splitlines() if l.strip()]
        if len(labels) < len(terms_list):
            labels += ["Andet"] * (len(terms_list) - len(labels))
        if len(labels) > len(terms_list):
            labels = labels[:len(terms_list)]
        overlaps = _has_word_overlap(labels)
        if overlaps:
            fix_spec = (
                "Ret disse kategorinavne, så ingen ord optræder i mere end én kategori. "
                "Bevar meningen. Svar igen som ren liste, én pr. linje."
            )
            prompt2 = fix_spec + "\n\n" + "\n".join(labels)
            resp2 = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content": prompt2}],
                temperature=0.1,
                max_tokens=256,
            )
            labels = [l.strip().strip("-•0123456789. ") for l in resp2.choices[0].message.content.splitlines() if l.strip()]
            if len(labels) < len(terms_list):
                labels += ["Andet"] * (len(terms_list) - len(labels))
            if len(labels) > len(terms_list):
                labels = labels[:len(terms_list)]
        return labels
    except Exception:
        return None

# --- Sammenlægning til target_n via agglomerativ klynge på center-vektorer ---
def merge_category_centers(centers, id2label, counts, target_n):
    """
    centers: np.array [K, F] rå-klyngecentre
    id2label: {cid: label}
    counts: antal svar pr. rå-klynge
    target_n: ønsket antal grupper for de 'flettelige' kategorier (uden Andet/Ved ikke)
    """
    K = centers.shape[0]
    if target_n >= K:
        return {i:i for i in range(K)}, [id2label[i] for i in range(K)]

    D = cosine_distances(centers)
    agg = AgglomerativeClustering(
        n_clusters=target_n,
        metric="precomputed",   # vigtigt i ny sklearn
        linkage="average",
    )
    groups = agg.fit_predict(D)

    group_labels = []
    for g in range(target_n):
        members = [i for i in range(K) if groups[i] == g]
        if not members:
            group_labels.append(f"Kategori {g+1}")
            continue
        best = max(members, key=lambda i: counts[i])
        group_labels.append(id2label[best])

    mapping = {i: groups[i] for i in range(K)}
    return mapping, group_labels

# -------------------------------------------------
# Hovedlogik
# -------------------------------------------------
if uploaded:
    log_event("file_upload", filename=getattr(uploaded, "name", "unknown"))

    with st.spinner("Indlæser og analyserer..."):
        df = read_excel_first_text_column(uploaded)
        texts = df["Svar"].tolist()

        # TF-IDF (ignorer tal/symboler i tokens via token_pattern)
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOP_WORDS),
            ngram_range=NGRAM_RANGE,
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            token_pattern=r"(?u)\b[^\W\d_]{2,}\b"  # kun bogstavsord (min 2 tegn)
        )
        X = vectorizer.fit_transform(texts)
        feature_names = np.array(vectorizer.get_feature_names_out())

        # Vælg K via silhouette; gem også labels for bedste K
        best_k, best_score, best_km, best_labels = None, -1, None, None
        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labs = km.fit_predict(X)
            try:
                score = silhouette_score(X, labs, metric="cosine", sample_size=min(10000, X.shape[0]))
            except Exception:
                score = -1
            if score > best_score:
                best_k, best_score, best_km, best_labels = k, score, km, labs

        km = best_km
        labels = best_labels
        centers = km.cluster_centers_
        centers_norm = normalize(centers, norm="l2", axis=1)
        X_norm = normalize(X, norm="l2", axis=1)

        # top-termer og antal pr. rå-klynge
        terms_per_cid = []
        counts_per_cid = []
        for cid in range(best_k):
            terms = top_terms_for_center(centers[cid], feature_names, topn=TOP_TERMS_PER_CAT)
            terms_per_cid.append(terms)
            counts_per_cid.append(int((labels == cid).sum()))

        # LLM foreslår entydige labels (globalt) — ellers fallback
        labels_llm = llm_name_categories_global(
            terms_per_cid, counts_per_cid,
            model=llm_model if use_llm else "gpt-4o-mini",
            temperature=float(llm_temperature)
        ) if use_llm else None

        # Entydige labels med normalisering + deduplikering
        categories_info = {}
        id2label = {}
        seen_norm = set()
        for cid in range(best_k):
            if labels_llm and cid < len(labels_llm) and labels_llm[cid]:
                raw_label = labels_llm[cid]
            else:
                raw_label = auto_label_from_terms(terms_per_cid[cid])
            label = normalize_label(raw_label)

            base = label
            suffix = 2
            while label.lower() in seen_norm:
                label = f"{base} {suffix}"
                suffix += 1

            seen_norm.add(label.lower())
            categories_info[label] = {
                "id": cid,
                "top_terms": terms_per_cid[cid],
                "count": counts_per_cid[cid]
            }
            id2label[cid] = label

        # --- Vis rå labels og top-termer
        st.subheader("Foreslåede kategorier (rå)")
        preview = pd.DataFrame({
            "Kategori": [id2label[cid] for cid in range(best_k)],
            "Antal svar (rå)": counts_per_cid,
            "Top-termer": [", ".join(terms_per_cid[cid][:5]) for cid in range(best_k)],
        })
        st.dataframe(preview, use_container_width=True)

        # --- Vælg samlet mål-antal inkl. 'Ved ikke' og 'Andet'
        target_total = st.slider(
            "Ønsket samlet antal kategorier (inkl. 'Ved ikke' og 'Andet')",
            min_value=5, max_value=20, value=min(10, max(5, best_k))
        )

        if st.button("Slå kategorier sammen til valgt antal"):
            # Vi fletter rå-kategorier ned til core = target_total - 2 (de 2 reserverede tilføjes bagefter)
            target_core = max(1, target_total - 2)

            mapping, core_labels = merge_category_centers(
                centers,
                {i: id2label[i] for i in range(best_k)},
                counts_per_cid,
                target_core
            )

            # Tilføj de to faste labels i slutlisten uden dubletter
            final_labels = list(core_labels)
            if not any(lbl.lower() == "ved ikke" for lbl in final_labels):
                final_labels.append("Ved ikke")
            if not any(lbl.lower() == "andet" for lbl in final_labels):
                final_labels.append("Andet")

            st.success(f"Slog {best_k} rå-kategorier sammen til {len(core_labels)} + 2 (Ved ikke, Andet) = {len(final_labels)} kategorier.")
            st.write("**Endelige kategorier:**")
            st.write(", ".join(final_labels))

            # --- Gruppér centre (gennemsnit) for core_labels og tildel multi-labels igen
            grp_centers = []
            for g in range(len(core_labels)):
                members = [i for i in range(best_k) if mapping[i] == g]
                if members:
                    grp_centers.append(np.mean(centers[members, :], axis=0))
                else:
                    grp_centers.append(np.zeros(centers.shape[1]))
            grp_centers = np.vstack(grp_centers)
            grp_centers_norm = normalize(grp_centers, norm="l2", axis=1)

            sims = X_norm @ grp_centers_norm.T  # [n_docs, len(core_labels)]

            # 'Ved ikke' detektor (meget simpel)
            def is_ved_ikke(text):
                t = text.strip().lower()
                return t in {"ved ikke", "ikke ved", "vedikke", "ukendt", "n/a", "na"} or t == ""

            # --- Bedre multi-label tildeling: relativ cut + "ord i teksten"-hint
            def top3_with_reserved(i_row):
                row = sims[i_row]

                # 1) eksplicit "Ved ikke"
                if is_ved_ikke(texts[i_row]):
                    return ["Ved ikke", "", ""], [1.0, np.nan, np.nan]

                # 2) relativ tærskel omkring bedste score + absolut bundgrænse
                idx_sorted = np.argsort(-row)
                best = float(row[idx_sorted[0]]) if idx_sorted.size else 0.0
                rel_cut = 0.88  # tag alt, der ligger inden for 88% af tophøjden
                abs_cut = 0.12  # men kræv også mindst denne absolutte værdi

                chosen = []
                for j in idx_sorted[:5]:
                    sc = float(row[j])
                    if sc >= abs_cut and (best == 0 or sc >= best * rel_cut):
                        chosen.append((j, sc))
                    if len(chosen) >= 3:
                        break

                # 3) ord i teksten = stærkt hint (force-inkludér op til 3 i alt)
                text_i = texts[i_row]
                for j in idx_sorted[:5]:
                    lab = core_labels[j]
                    sc = float(row[j])
                    if contains_category_word(text_i, lab) and (j, sc) not in chosen:
                        chosen.append((j, sc))
                    if len(chosen) >= 3:
                        break

                if not chosen:
                    return ["Andet", "", ""], [0.0, np.nan, np.nan]

                labs = [core_labels[j] for j, _ in chosen[:3]]
                scs  = [float(s) for _, s in chosen[:3]]
                while len(labs) < 3:
                    labs.append(""); scs.append(np.nan)
                return labs, scs

            cats_1, cats_2, cats_3 = [], [], []
            sc_1, sc_2, sc_3 = [], [], []
            for i in range(sims.shape[0]):
                labs3, scs3 = top3_with_reserved(i)
                cats_1.append(labs3[0]); cats_2.append(labs3[1]); cats_3.append(labs3[2])
                sc_1.append(scs3[0]);   sc_2.append(scs3[1]);   sc_3.append(scs3[2])

            df_out = pd.DataFrame({
                "Svar": texts,
                "Kategori_1": cats_1,
                "Kategori_2": cats_2,
                "Kategori_3": cats_3,
                "Score_1": sc_1,
                "Score_2": sc_2,
                "Score_3": sc_3,
            })

            # Analyse (efter sammenlægning)
            def build_analysis_sheet(df_data, labels_final):
                total = len(df_data)
                counts = Counter(df_data["Kategori_1"].fillna("Andet"))
                rows = []
                for cat in labels_final:
                    cnt = counts.get(cat, 0)
                    pct = (cnt / total * 100.0) if total else 0.0
                    examples = (
                        df_data.loc[df_data["Kategori_1"] == cat]
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
                        "Citater 1": examples[0],
                        "Citater 2": examples[1],
                        "Citater 3": examples[2],
                    })
                rows = sorted(rows, key=lambda r: r["Antal"], reverse=True)
                return pd.DataFrame(rows)

            final_labels = list(core_labels)
            if not any(lbl.lower() == "ved ikke" for lbl in final_labels):
                final_labels.append("Ved ikke")
            if not any(lbl.lower() == "andet" for lbl in final_labels):
                final_labels.append("Andet")

            analysis_df = build_analysis_sheet(df_out, final_labels)

            st.subheader("Eksempel (efter sammenlægning)")
            st.dataframe(df_out.head(20), use_container_width=True)
            st.subheader("Analyse (efter sammenlægning)")
            st.dataframe(analysis_df, use_container_width=True)

            # Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_out.to_excel(writer, sheet_name="Data", index=False)
                meta = pd.DataFrame([{
                    "Dato": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                    "Antal svar": len(df_out),
                    "Rå-kategorier": best_k,
                    "Kategorier efter sammenlægning (inkl. Andet/Ved ikke)": len(final_labels),
                }])
                meta.to_excel(writer, sheet_name="Analyse", index=False, startrow=0)
                analysis_df.to_excel(writer, sheet_name="Analyse", index=False, startrow=3)
            buffer.seek(0)
            st.download_button(
                "Download Excel (efter sammenlægning)",
                data=buffer,
                file_name="aabne_svar_kategoriseret_merged.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=lambda: log_event("download_after_merge", rows=len(df_out), k_raw=best_k, k_final=len(final_labels))
            )

st.divider()
st.subheader("Session-log")
download_audit_widget()
