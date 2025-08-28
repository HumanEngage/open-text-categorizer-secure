# app.py — Streamlit: Ren LLM-baseret analyse af åbne svar (med stram kategorifinding)
# - Password via Secrets: APP_PASSWORD="din-kode"
# - Kræver OpenAI: OPENAI_API_KEY="sk-..."
# - Flow:
#   1) LLM foreslår entydige kategorier (1–3 ord, uden ord-overlap) — 10–16 stk
#   2) Du vælger samlet antal (5–20); LLM slår dem sammen til målet
#   3) LLM klassificerer ALLE svar (op til 3 labels pr. svar, inkl. "Ved ikke"/"Andet")
#   4) Download Excel

import os
import io
import re
import json
from datetime import datetime
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

# ---------------------------------------------
# Page config & adgangskontrol
# ---------------------------------------------
st.set_page_config(page_title="Åbne svar → Ren LLM-kategorisering", layout="wide")

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

# ---------------------------------------------
# UI top
# ---------------------------------------------
st.title("Åbne besvarelser → Ren LLM-kategorisering")
st.caption("LLM finder kategorier + klassificerer alle svar (multi-label). Korte, entydige labels uden ord-overlap. ‘Ved ikke’ og ‘Andet’ er altid mulige.")

uploaded = st.file_uploader("Upload Excel (.xlsx) – første kolonne skal indeholde de åbne svar", type=["xlsx"])

# LLM indstillinger
api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    model_labels = st.text_input("Model til navngivning/merging", value="gpt-4o-mini")
with col_b:
    model_assign = st.text_input("Model til klassifikation af svar", value="gpt-4o-mini")
with col_c:
    temperature_labels = st.slider("Temperature (labels/merging)", 0.0, 1.0, 0.2, 0.05)

st.info(f"LLM-status: {'🔑 Nøgle fundet' if api_key_present else '⛔ Ingen OPENAI_API_KEY i Secrets'}")

def quick_llm_ping(model="gpt-4o-mini"):
    try:
        from openai import OpenAI
        if not api_key_present:
            return False, "Ingen OPENAI_API_KEY i miljøet."
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":"Svar kun med ordet: OK"}],
            max_tokens=2,
            temperature=0.0,
        )
        return True, resp.choices[0].message.content.strip()
    except Exception as e:
        return False, str(e)

if st.button("Test LLM-forbindelse"):
    ok, msg = quick_llm_ping(model_labels)
    if ok:
        st.success(f"LLM-kald lykkedes: {msg}")
    else:
        st.error(f"LLM-kald fejlede: {msg}")

# ---------------------------------------------
# Hjælpere / konstanter
# ---------------------------------------------
RESERVED = ["Ved ikke", "Andet"]

# Anti-meta og fallback
FORBIDDEN_LABEL_WORDS = {
    "json","liste","list","kategori","kategorier","kommentar","kommentarer",
    "data","datasæt","dataset","output","returner","returnér","model","openai",
    "tekst","svar","anonym","felter","kolonne","excel"
}
BASE_FALLBACK_LABELS = [
    "Pris","Hastighed","Dækning","Kundeservice","Stabilitet","Funktioner",
    "Installation","Brugervenlighed","Kommunikation","Transparens",
    "Værdi for pengene","Vilkår og binding","Support svartid","Tilgængelighed",
    "Kvalitet","Fleksibilitet"
]

STOPWORDS_DA = {
    "og","i","jeg","det","at","en","den","til","er","som","på","de","med","der",
    "har","for","af","ikke","et","men","vi","kan","om","så","ud","over","sig",
    "fra","bliver","eller","hvad","hvordan","hvorfor","når","man","noget",
    "meget","mere","mest","alle","alt","andre","også","kun","bare","ind","op",
    "ned","hos","hen","derfor","fordi","kunne","skulle","ville","bl.a.","fx","f.eks."
}

def read_first_text_column(file) -> pd.DataFrame:
    df = pd.read_excel(file)
    if df.shape[1] == 0:
        raise ValueError("Excel-arket er tomt.")
    candidates = [c for c in df.columns if str(c).strip().lower() in {"svar","tekst","text","response","kommentar"}]
    col = candidates[0] if candidates else df.columns[0]
    s = df[col].astype(str).fillna("").map(lambda x: x.strip())
    s = s[s.str.len() > 0]
    return pd.DataFrame({"Svar": s})

def sample_texts_for_prompt(texts: List[str], max_items: int = 400) -> List[str]:
    if len(texts) <= max_items:
        return texts
    step = max(1, len(texts) // max_items)
    return [texts[i] for i in range(0, len(texts), step)][:max_items]

def parse_json_lines_list(content: str) -> List[str]:
    txt = content.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return [str(x).strip() for x in obj["labels"] if str(x).strip()]
    except Exception:
        pass
    lines = [l.strip().strip("-•0123456789. ").strip() for l in txt.splitlines() if l.strip()]
    return lines

def normalize_label(label: str) -> str:
    s = re.sub(r"[^A-Za-zÆØÅæøå\s]", " ", label).strip().lower()
    words = [w for w in re.split(r"\s+", s) if w]
    def _strip_dk(w):
        for suf in ["erne","enes","ende","tene","eren","er","en","et","e"]:
            if w.endswith(suf) and len(w) > len(suf) + 1:
                return w[:-len(suf)]
        return w
    words = [_strip_dk(w) for w in words][:3]
    if not words:
        return "Andet"
    out = " ".join(words).capitalize()
    return out

def dedup_and_enforce_unique_words(labels: List[str]) -> List[str]:
    cleaned = [normalize_label(l) for l in labels if l]
    cleaned = [c for c in cleaned if c.lower() not in {"ved ikke","andet"}]
    cleaned = [c for c in cleaned if all(w.lower() not in FORBIDDEN_LABEL_WORDS for w in c.split())]
    seen, out = set(), []
    for lab in cleaned:
        if lab.lower() not in seen:
            out.append(lab); seen.add(lab.lower())
    first_seen, final = set(), []
    for lab in out:
        first = lab.split()[0].lower()
        if first in first_seen:
            continue
        first_seen.add(first)
        final.append(lab)
    return final

def _tokenize_clean(s: str):
    toks = [t for t in re.findall(r"[A-Za-zÆØÅæøå]+", s.lower()) if len(t) >= 3]
    return [t for t in toks if t not in STOPWORDS_DA]

def build_term_hints(texts, max_unigrams=30, max_bigrams=20):
    from collections import Counter
    uni, bi = Counter(), Counter()
    for t in texts:
        w = _tokenize_clean(t)
        uni.update(w)
        for i in range(len(w)-1):
            if w[i] in STOPWORDS_DA or w[i+1] in STOPWORDS_DA:
                continue
            bi.update([w[i] + " " + w[i+1]])
    top_uni = [w for w,_ in uni.most_common(max_unigrams)]
    top_bi  = [b for b,_ in bi.most_common(max_bigrams)]
    return top_uni, top_bi

# ---------------------------------------------
# OpenAI wrappers
# ---------------------------------------------
def openai_client():
    from openai import OpenAI
    if not api_key_present:
        raise RuntimeError("OPENAI_API_KEY mangler i Secrets.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_propose_categories(texts: List[str], model: str, temperature: float) -> List[str]:
    """
    Foreslå 10–16 korte, entydige kategorier (uden 'Ved ikke'/'Andet') fra et udsnit af datasættet.
    Stram JSON-mode, forbudte ord, hints fra hyppige ord/fraser, og håndhævet min/max.
    """
    client = openai_client()
    sample = sample_texts_for_prompt(texts, max_items=400)
    uni, bi = build_term_hints(sample, max_unigrams=30, max_bigrams=20)

    N_MIN, N_MAX = 10, 16

    prompt = (
        "Du er en dansktalende analyseassistent. Du får et udsnit af åbne kommentarer.\n"
        f"OPGAVE: Foreslå mellem {N_MIN} og {N_MAX} KORTE, ENTydige kategorier (1–3 ord).\n"
        "- Brug almindelige tema-ord; ingen tal/specialtegn/jargon.\n"
        "- INGEN labels må dele samme ord (fx både 'Pris' og 'Høj pris' er ikke ok).\n"
        "- Brug ALDRIG meta-ord som: json, liste, kategorier, data, model, output, svar, tekst.\n"
        "- MEDTAG IKKE 'Ved ikke' eller 'Andet' (de håndteres separat).\n"
        "- Vælg labels der kan genbruges på tværs af mange typer feedback.\n"
        "- Returnér KUN en JSON med formen: {\"labels\": [\"...\"]}\n\n"
        "Hyppige ord (hint): " + ", ".join(uni[:30]) + "\n"
        "Hyppige fraser (hint): " + ", ".join(bi[:20]) + "\n\n"
        "Udsnit af kommentarer:\n" + "\n".join(f"- {t[:300]}" for t in sample[:150]) + "\n"
    )

    labels = []
    try:
        # JSON-mode
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content": prompt}],
            temperature=temperature,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            labels = [str(x) for x in obj["labels"]]
    except Exception:
        # fallback uden JSON-mode
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content": prompt + "\nReturnér kun JSON som beskrevet."}],
            temperature=temperature,
            max_tokens=700,
        )
        labels = parse_json_lines_list(resp.choices[0].message.content)

    labels = dedup_and_enforce_unique_words(labels)

    # Håndhæv MIN/MAX og fyld op fra basislisten, hvis nødvendigt
    if len(labels) < 10:
        for cand in BASE_FALLBACK_LABELS:
            if cand not in labels:
                labels.append(cand)
            if len(labels) >= 10:
                break
    if len(labels) > 16:
        labels = labels[:16]

    # Endeligt filter mod forbudte ord
    labels = [l for l in labels if all(w.lower() not in FORBIDDEN_LABEL_WORDS for w in l.split())]
    return labels

def llm_merge_categories(labels: List[str], target_core: int, model: str, temperature: float) -> List[str]:
    """
    Slå labels sammen til target_core entydige kategorier (uden 'Ved ikke'/'Andet').
    """
    client = openai_client()
    prompt = (
        "Du får en liste af kategorier. Slå dem sammen til præcis "
        f"{target_core} entydige kategorier (1–3 danske ord), uden at dele ord mellem labels.\n"
        "- Undgå tal/specialtegn og meta-ord (json, liste, data, osv.).\n"
        "- Returnér KUN en JSON med formen: {\"labels\": [\"...\"]}\n\n"
        f"Indgående kategorier:\n{json.dumps(labels, ensure_ascii=False)}\n"
    )
    merged = []
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content": prompt}],
            temperature=temperature,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            merged = [str(x) for x in obj["labels"]]
    except Exception:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content": prompt + "\nReturnér kun JSON som beskrevet."}],
            temperature=temperature,
            max_tokens=700,
        )
        merged = parse_json_lines_list(resp.choices[0].message.content)

    merged = dedup_and_enforce_unique_words(merged)

    # Sikr korrekt antal (pad/trunc)
    if len(merged) < target_core:
        for cand in BASE_FALLBACK_LABELS:
            if cand not in merged:
                merged.append(cand)
            if len(merged) >= target_core:
                break
    if len(merged) > target_core:
        merged = merged[:target_core]
    return merged

def openai_client():
    from openai import OpenAI
    if not api_key_present:
        raise RuntimeError("OPENAI_API_KEY mangler i Secrets.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def llm_assign_batch(categories: List[str], texts: List[str], model: str) -> List[List[str]]:
    """
    Klassificér en batch af tekster. Returnér liste af lister (op til 3 labels pr. tekst).
    Kategorienavne skal vælges fra categories + RESERVED.
    """
    client = openai_client()
    sys = ("Du er en hjælpsom, deterministisk klassifikations-assistent. Giv kun JSON som svar.")
    cat_list = categories + RESERVED
    instruct = {
        "task": "Multi-label klassifikation af danske kommentarer",
        "rules": [
            "Vælg 1–3 labels fra 'categories' for hver kommentar",
            "Hvis kommentaren udtrykkeligt svarer 'ved ikke'/'ukendt'/tom, brug 'Ved ikke'",
            "Hvis ingen label passer rimeligt, brug 'Andet'",
            "Returnér KUN JSON: en liste hvor hver post er en liste med 1–3 kategorier fra 'categories'",
        ],
        "categories": cat_list
    }
    joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    user = (
        json.dumps(instruct, ensure_ascii=False) +
        "\n\nHer er kommentarerne (nummereret):\n" + joined +
        "\n\nReturnér kun JSON-listen: fx [[\"Pris\",\"Hastighed\"],[\"Dækning\"], ...]"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content": sys}, {"role":"user","content": user}],
        temperature=0.0,
        max_tokens=4000,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            out = []
            for item in obj:
                if isinstance(item, list):
                    labs = [str(x) for x in item if str(x).strip()]
                    labs = [l for l in labs if l in cat_list]
                    if not labs:
                        labs = ["Andet"]
                    out.append(labs[:3])
                else:
                    out.append(["Andet"])
            if len(out) < len(texts):
                out += [["Andet"]] * (len(texts) - len(out))
            if len(out) > len(texts):
                out = out[:len(texts)]
            return out
    except Exception:
        pass
    return [["Andet"] for _ in texts]

# ---------------------------------------------
# Hovedflow
# ---------------------------------------------
if uploaded:
    if not api_key_present:
        st.error("Du skal sætte OPENAI_API_KEY i Secrets for at køre ren LLM-analyse.")
        st.stop()

    df = read_first_text_column(uploaded)
    texts_all = df["Svar"].tolist()
    log_event("file_upload", rows=len(texts_all))

    st.write(f"Antal svar: **{len(texts_all)}**")

    # 1) Foreslå rå kategorier
    st.subheader("1) LLM foreslår kategorier")
    if st.button("Foreslå kategorier (LLM)"):
        with st.spinner("Foreslår kategorier..."):
            labels_raw = llm_propose_categories(texts_all, model_labels, temperature_labels)
            st.session_state["labels_raw"] = labels_raw
            log_event("labels_proposed", count=len(labels_raw))
    labels_raw = st.session_state.get("labels_raw", None)
    if labels_raw:
        st.success(f"Foreslået: {len(labels_raw)} kategorier (uden 'Ved ikke'/'Andet').")
        st.dataframe(pd.DataFrame({"Kategorier (rå)": labels_raw}), use_container_width=True)

    # 2) Vælg samlet mål-antal (inkl. 'Ved ikke'/'Andet') og merge
    st.subheader("2) Vælg samlet antal og slå sammen")
    target_total = st.slider("Samlet antal kategorier (inkl. 'Ved ikke' og 'Andet')", 5, 20, 10)
    if labels_raw and st.button("Slå sammen til valgt antal (LLM)"):
        with st.spinner("Slår sammen..."):
            target_core = max(1, target_total - 2)
            merged_core = llm_merge_categories(labels_raw, target_core, model_labels, temperature_labels)
            final_labels = list(merged_core)
            if "Ved ikke" not in [x.lower() for x in final_labels]:
                final_labels.append("Ved ikke")
            if "Andet" not in [x.lower() for x in final_labels]:
                final_labels.append("Andet")
            st.session_state["final_labels"] = final_labels
            log_event("labels_merged", core=len(merged_core), total=len(final_labels))

    final_labels = st.session_state.get("final_labels", None)
    if final_labels:
        st.success(f"Endelige kategorier ({len(final_labels)}):")
        st.write(", ".join(final_labels))

    # 3) Klassificér alle svar (LLM i batches)
    st.subheader("3) Klassificér alle svar (LLM)")
    batch_size = st.number_input("Batch-størrelse (flere = hurtigere, men større prompt)", min_value=10, max_value=200, value=50, step=10)
    if final_labels and st.button("Kør klassifikation på alle svar"):
        with st.spinner("Klassificerer alle svar..."):
            assignments: List[List[str]] = []
            n = len(texts_all)
            for i in range(0, n, batch_size):
                batch = texts_all[i:i+batch_size]
                preds = llm_assign_batch(final_labels, batch, model_assign)
                assignments.extend(preds)
            cats1, cats2, cats3 = [], [], []
            for labs in assignments:
                labs = labs[:3] + ["", "", ""]
                cats1.append(labs[0]); cats2.append(labs[1]); cats3.append(labs[2])
            df_out = pd.DataFrame({
                "Svar": texts_all,
                "Kategori_1": cats1,
                "Kategori_2": cats2,
                "Kategori_3": cats3,
            })
            st.session_state["df_out"] = df_out
            log_event("assigned_all", rows=len(df_out))

    df_out: pd.DataFrame = st.session_state.get("df_out", None)
    if df_out is not None:
        st.subheader("Eksempel på resultat (første 30 rækker)")
        st.dataframe(df_out.head(30), use_container_width=True)

        # Analyse
        st.subheader("Analyse")
        def build_analysis(df_data: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
            total = len(df_data)
            counts = df_data["Kategori_1"].value_counts(dropna=False).to_dict()
            rows = []
            for cat in labels:
                cnt = int(counts.get(cat, 0))
                pct = round((cnt / total * 100.0), 1) if total else 0.0
                ex = df_data.loc[df_data["Kategori_1"] == cat, "Svar"].head(3).tolist()
                ex += [""] * (3 - len(ex))
                rows.append({
                    "Kategori": cat,
                    "Antal": cnt,
                    "Andel %": pct,
                    "Citater 1": ex[0],
                    "Citater 2": ex[1],
                    "Citater 3": ex[2],
                })
            return pd.DataFrame(rows).sort_values("Antal", ascending=False)

        analysis_df = build_analysis(df_out, final_labels or [])
        st.dataframe(analysis_df, use_container_width=True)

        # Download Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Data", index=False)
            meta = pd.DataFrame([{
                "Dato": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "Antal svar": len(df_out),
                "Antal kategorier": len(final_labels or []),
            }])
            meta.to_excel(writer, sheet_name="Analyse", index=False, startrow=0)
            analysis_df.to_excel(writer, sheet_name="Analyse", index=False, startrow=3)
        buffer.seek(0)
        st.download_button(
            "Download Excel (LLM-kategoriseret)",
            data=buffer,
            file_name="aabne_svar_LLM.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click=lambda: log_event("download", rows=len(df_out), k=len(final_labels or []))
        )

st.divider()
st.subheader("Session-log")
download_audit_widget()
