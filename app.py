# app.py — Streamlit: Ren LLM-baseret kategorisering af åbne svar
# Kræver Secrets:
#   APP_PASSWORD="din-kode"
#   OPENAI_API_KEY="sk-..."
#
# Flow:
# 1) LLM foreslår 10–16 entydige kategorier (1–3 ord, ingen ord-overlap)
# 2) Slå sammen til valgt antal (5–20) — uden 'Ved ikke'/'Andet'
# 3) (NYT) Redigér kategorier manuelt (tekstboks)
# 4) LLM klassificerer ALLE svar — ID-baseret, multi-label (1–3 pr. svar), med fuzzy fallback
# 5) Download Excel

import os, io, re, json
from datetime import datetime
from typing import List

import streamlit as st
import pandas as pd

# ---------------- Page & auth ----------------
st.set_page_config(page_title="Åbne svar → LLM-kategorisering (secure)", layout="wide")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", None)

def check_auth():
    if APP_PASSWORD is None:
        st.warning("⚠️ Ingen APP_PASSWORD sat i Secrets — appen er åben for alle med link.")
        return True
    if st.session_state.get("auth_ok"):
        return True
    st.title("🔒 Adgang påkrævet")
    pwd = st.text_input("Adgangskode", type="password")
    if st.button("Log ind"):
        if pwd == APP_PASSWORD:
            st.session_state["auth_ok"] = True
            st.rerun()
        else:
            st.error("Forkert adgangskode")
    return False

if not check_auth():
    st.stop()

def log_event(action, **kwargs):
    st.session_state.setdefault("audit", [])
    row = {"ts": datetime.utcnow().isoformat(), "action": action}
    row.update(kwargs)
    st.session_state["audit"].append(row)

def download_audit_widget():
    if not st.session_state.get("audit"):
        return
    df_log = pd.DataFrame(st.session_state["audit"])
    st.download_button("Download session-log (CSV)",
                       data=df_log.to_csv(index=False).encode("utf-8"),
                       file_name="session_audit_log.csv", mime="text/csv")

# ---------------- UI top ----------------
st.title("Åbne besvarelser → Ren LLM-kategorisering")
st.caption("LLM finder kategorier og klassificerer alle svar (multi-label). Labels er korte og entydige. ‘Ved ikke’ og ‘Andet’ er altid mulige.")

uploaded = st.file_uploader("Upload Excel (.xlsx) — første kolonne bør være de åbne svar", type=["xlsx"])

api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
c1, c2, c3 = st.columns([1,1,1])
with c1: model_labels  = st.text_input("Model (labels/merge)", value="gpt-4o-mini")
with c2: model_assign  = st.text_input("Model (klassifikation)", value="gpt-4o-mini")
with c3: temperature_labels = st.slider("Temperature (labels/merge)", 0.0, 1.0, 0.2, 0.05)

st.info(f"LLM-status: {'🔑 Nøgle fundet' if api_key_present else '⛔ OPENAI_API_KEY mangler i Secrets'}")

# ---------------- Hjælpere & konstanter ----------------
RESERVED = ["Ved ikke", "Andet"]

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
    if df.shape[1] == 0: raise ValueError("Excel-arket er tomt.")
    candidates = [c for c in df.columns if str(c).strip().lower() in {"svar","tekst","text","response","kommentar"}]
    col = candidates[0] if candidates else df.columns[0]
    s = df[col].astype(str).fillna("").map(lambda x: x.strip())
    s = s[s.str.len() > 0]
    return pd.DataFrame({"Svar": s})

def sample_texts_for_prompt(texts: List[str], max_items=400) -> List[str]:
    if len(texts) <= max_items: return texts
    step = max(1, len(texts)//max_items)
    return [texts[i] for i in range(0, len(texts), step)][:max_items]

def parse_json_lines_list(content: str) -> List[str]:
    txt = content.strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, list): return [str(x).strip() for x in obj if str(x).strip()]
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            return [str(x).strip() for x in obj["labels"] if str(x).strip()]
    except Exception:
        pass
    return [l.strip().strip("-•0123456789. ").strip() for l in txt.splitlines() if l.strip()]

def normalize_label(label: str) -> str:
    s = re.sub(r"[^A-Za-zÆØÅæøå\s]", " ", label).strip().lower()
    words = [w for w in re.split(r"\s+", s) if w]
    def _strip_dk(w):
        for suf in ["erne","enes","ende","tene","eren","er","en","et","e"]:
            if w.endswith(suf) and len(w) > len(suf) + 1: return w[:-len(suf)]
        return w
    words = [_strip_dk(w) for w in words][:3]
    return "Andet" if not words else " ".join(words).capitalize()

def dedup_and_enforce_unique_words(labels: List[str]) -> List[str]:
    cleaned = [normalize_label(l) for l in labels if l]
    cleaned = [c for c in cleaned if c.lower() not in {"ved ikke","andet"}]
    cleaned = [c for c in cleaned if all(w.lower() not in FORBIDDEN_LABEL_WORDS for w in c.split())]
    seen, out = set(), []
    for lab in cleaned:
        if lab.lower() not in seen: out.append(lab); seen.add(lab.lower())
    first_seen, final = set(), []
    for lab in out:
        first = lab.split()[0].lower()
        if first in first_seen: continue
        first_seen.add(first); final.append(lab)
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
            if w[i] in STOPWORDS_DA or w[i+1] in STOPWORDS_DA: continue
            bi.update([w[i] + " " + w[i+1]])
    return [w for w,_ in uni.most_common(max_unigrams)], [b for b,_ in bi.most_common(max_bigrams)]

def openai_client():
    from openai import OpenAI
    if not api_key_present: raise RuntimeError("OPENAI_API_KEY mangler i Secrets.")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# --------- LLM: foreslå kategorier ---------
def llm_propose_categories(texts: List[str], model: str, temperature: float) -> List[str]:
    client = openai_client()
    sample = sample_texts_for_prompt(texts, 400)
    uni, bi = build_term_hints(sample)

    N_MIN, N_MAX = 10, 16
    prompt = (
        "Du er en dansktalende analyseassistent. Du får et udsnit af åbne kommentarer.\n"
        f"OPGAVE: Foreslå mellem {N_MIN} og {N_MAX} KORTE, ENTydige kategorier (1–3 ord).\n"
        "- Brug almindelige tema-ord; ingen tal/specialtegn/jargon.\n"
        "- INGEN labels må dele samme ord (fx både 'Pris' og 'Høj pris' er ikke ok).\n"
        "- Brug ALDRIG meta-ord som: json, liste, kategorier, data, model, output, svar, tekst.\n"
        "- MEDTAG IKKE 'Ved ikke' eller 'Andet' (de håndteres separat).\n"
        "- Returnér KUN en JSON med formen: {\"labels\": [\"...\"]}\n\n"
        "Hyppige ord (hint): " + ", ".join(uni[:30]) + "\n"
        "Hyppige fraser (hint): " + ", ".join(bi[:20]) + "\n\n"
        "Udsnit af kommentarer:\n" + "\n".join(f"- {t[:300]}" for t in sample[:150]) + "\n"
    )

    labels = []
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content": prompt}],
            temperature=temperature, max_tokens=700,
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            labels = [str(x) for x in obj["labels"]]
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content": prompt + "\nReturnér kun JSON som beskrevet."}],
            temperature=temperature, max_tokens=700,
        )
        labels = parse_json_lines_list(resp.choices[0].message.content)

    labels = dedup_and_enforce_unique_words(labels)
    if len(labels) < 10:
        for cand in BASE_FALLBACK_LABELS:
            if cand not in labels:
                labels.append(cand)
            if len(labels) >= 10: break
    if len(labels) > 16: labels = labels[:16]
    labels = [l for l in labels if all(w.lower() not in FORBIDDEN_LABEL_WORDS for w in l.split())]
    return labels

# --------- LLM: merge kategorier til target ---------
def llm_merge_categories(labels: List[str], target_core: int, model: str, temperature: float) -> List[str]:
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
            model=model, messages=[{"role":"user","content": prompt}],
            temperature=temperature, max_tokens=700,
            response_format={"type": "json_object"},
        )
        obj = json.loads(resp.choices[0].message.content)
        if isinstance(obj, dict) and isinstance(obj.get("labels"), list):
            merged = [str(x) for x in obj["labels"]]
    except Exception:
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content": prompt + "\nReturnér kun JSON som beskrevet."}],
            temperature=temperature, max_tokens=700,
        )
        merged = parse_json_lines_list(resp.choices[0].message.content)

    merged = dedup_and_enforce_unique_words(merged)
    if len(merged) < target_core:
        for cand in BASE_FALLBACK_LABELS:
            if cand not in merged:
                merged.append(cand)
            if len(merged) >= target_core: break
    if len(merged) > target_core: merged = merged[:target_core]
    return merged

# --------- (NY) Manuelt rediger labels ---------
def apply_label_edit(text_block: str) -> List[str]:
    raw_lines = [l.strip() for l in text_block.splitlines() if l.strip()]
    return dedup_and_enforce_unique_words(raw_lines)

# --------- LLM: klassifikation m/ ID'er + multi-label-demo + fuzzy fallback ---------
def llm_assign_batch(categories: List[str], texts: List[str], model: str) -> List[List[str]]:
    client = openai_client()

    # byg id->label inkl. reserverede
    cats = [{"id": i + 1, "label": lab} for i, lab in enumerate(categories + RESERVED)]
    id2label = {c["id"]: c["label"] for c in cats}
    label_lowers = {lbl.lower(): lbl for lbl in id2label.values()}

    # simple helper: map tekst -> labels via substring/token match
    def fuzzy_labels_from_text(s: str) -> List[str]:
        s_low = s.lower()
        labs = []
        # split på typiske separatorer
        parts = re.split(r"[,/]| og | & | \+ ", s_low)
        for p in parts:
            p = p.strip()
            for low, orig in label_lowers.items():
                if low in {"ved ikke","andet"}:  # håndteres særskilt af regler
                    continue
                if low in p:
                    labs.append(orig)
        # dedup og maks 3
        out = []
        for l in labs:
            if l not in out: out.append(l)
            if len(out) >= 3: break
        return out

    sys = (
        "Du er en deterministisk dansk klassifikations-assistent. "
        "Svar KUN med gyldig JSON der kan parses."
    )

    # Multi-label eksempel (pris + hastighed)
    pris_id = next((c['id'] for c in cats if c['label'].lower() == "pris"), None)
    hast_id = next((c['id'] for c in cats if c['label'].lower() == "hastighed"), None)
    ved_ikke_id = next(c['id'] for c in cats if c['label'] == "Ved ikke")
    demo_user = {
        "task": "Multi-label klassifikation af danske kommentarer",
        "rules": [
            "For hver kommentar: vælg 1–3 KATEGORI-ID'er fra 'categories'.",
            "Hvis kommentaren er tom/‘ved ikke’/uklar → brug ID'et for 'Ved ikke'.",
            "Hvis ingen kategori passer rimeligt → brug ID'et for 'Andet'.",
            "Returnér KUN JSON-objekt: {\"preds\": [[id,id],[id], ...]} (samme længde som antal kommentarer)."
        ],
        "categories": cats,
        "example": {
            "comment": "Pris og hastighed er vigtige",
            "expected_ids": [x for x in [pris_id, hast_id] if x]
        }
    }

    # Rigtig batch
    joined = "\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    real_user = {
        "task": demo_user["task"],
        "rules": demo_user["rules"],
        "categories": cats,
        "comments_numbered": joined
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content": sys},
            {"role":"user","content": json.dumps(demo_user, ensure_ascii=False)},
            {"role":"assistant","content": json.dumps({"preds": [[x for x in [pris_id, hast_id] if x], [ved_ikke_id]]}, ensure_ascii=False)},
            {"role":"user","content": json.dumps(real_user, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    try:
        obj = json.loads(resp.choices[0].message.content.strip())
    except Exception:
        return [["Andet"] for _ in texts]

    pred_ids = obj.get("preds") if isinstance(obj, dict) else None
    if not isinstance(pred_ids, list):
        return [["Andet"] for _ in texts]

    out = []
    for i, item in enumerate(pred_ids[:len(texts)]):
        labs = []
        # Primært: forventer liste af heltals-ID'er
        if isinstance(item, list):
            for v in item[:3]:
                if isinstance(v, int) and v in id2label:
                    labs.append(id2label[v])
                elif isinstance(v, str):  # Fallback: modellen skrev tekst → forsøg fuzzy mapping
                    labs += fuzzy_labels_from_text(v)
        # Hvis intet, prøv at se om modellen kom til at skrive en enkelt streng
        if not labs and isinstance(item, str):
            labs = fuzzy_labels_from_text(item)
        # Sidste fallback
        if not labs:
            # hvis kommentaren ligner "ved ikke"
            if re.fullmatch(r"\s*(ved\s*ikke|ukendt|n/?a|na)?\s*", texts[i].lower()):
                labs = ["Ved ikke"]
            else:
                labs = ["Andet"]
        # dedup + max 3
        uniq = []
        for l in labs:
            if l not in uniq:
                uniq.append(l)
            if len(uniq) >= 3: break
        out.append(uniq)

    while len(out) < len(texts):
        out.append(["Andet"])
    return out

# ---------------- Hovedflow ----------------
if uploaded:
    if not api_key_present:
        st.error("Sæt OPENAI_API_KEY i Secrets for at køre ren LLM-analyse.")
        st.stop()

    df = read_first_text_column(uploaded)
    texts_all = df["Svar"].tolist()
    log_event("file_upload", rows=len(texts_all))
    st.write(f"Antal svar: **{len(texts_all)}**")

    # 1) Foreslå kategorier
    st.subheader("1) LLM foreslår kategorier")
    if st.button("Foreslå kategorier (LLM)"):
        with st.spinner("Foreslår kategorier..."):
            labels_raw = llm_propose_categories(texts_all, model_labels, temperature_labels)
            st.session_state["labels_raw"] = labels_raw
            log_event("labels_proposed", count=len(labels_raw))
    labels_raw = st.session_state.get("labels_raw")
    if labels_raw:
        st.success(f"Foreslået: {len(labels_raw)} kategorier (uden 'Ved ikke'/'Andet').")
        st.dataframe(pd.DataFrame({"Kategorier (rå)": labels_raw}), use_container_width=True)

    # 2) Slå sammen til valgt antal
    st.subheader("2) Vælg samlet antal og slå sammen")
    target_total = st.slider("Samlet antal kategorier (inkl. 'Ved ikke' og 'Andet')", 5, 20, 10)
    if labels_raw and st.button("Slå sammen til valgt antal (LLM)"):
        with st.spinner("Slår sammen..."):
            core = max(1, target_total - 2)
            merged = llm_merge_categories(labels_raw, core, model_labels, temperature_labels)
            final_labels = list(merged)
            # Tilføj reserverede
            if "Ved ikke" not in [l for l in final_labels]: final_labels.append("Ved ikke")
            if "Andet" not in [l for l in final_labels]: final_labels.append("Andet")
            st.session_state["final_labels"] = final_labels
            log_event("labels_merged", core=len(merged), total=len(final_labels))

    # 2b) (NYT) Manuelt redigér labels
    final_labels = st.session_state.get("final_labels")
    if final_labels:
        st.success(f"Endelige kategorier ({len(final_labels)}):")
        st.write(", ".join(final_labels))

        st.subheader("2b) Ret kategorier manuelt (valgfrit)")
        st.caption("Én kategori pr. linje. Korte, entydige ord (1–3). ‘Ved ikke’ og ‘Andet’ tilføjes automatisk.")
        non_reserved = [l for l in final_labels if l not in RESERVED]
        edited_text = st.text_area("Redigér kategorier", value="\n".join(non_reserved), height=220)
        colx, coly = st.columns([1,2])
        with colx:
            if st.button("Gem redigering"):
                cleaned = apply_label_edit(edited_text)
                # håndhæv 5–20 inkl. reserverede
                total_with_reserved = len(cleaned) + 2
                if total_with_reserved < 5:
                    st.warning("For få kategorier — fylder op fra basislisten til min. 5 i alt.")
                    for cand in BASE_FALLBACK_LABELS:
                        if cand not in cleaned:
                            cleaned.append(cand)
                        if len(cleaned) + 2 >= 5: break
                if total_with_reserved > 20:
                    st.warning("For mange kategorier — trimmer til maks 20 i alt.")
                    cleaned = cleaned[:18]  # plads til 2 reserverede
                new_final = list(cleaned)
                if "Ved ikke" not in new_final: new_final.append("Ved ikke")
                if "Andet" not in new_final: new_final.append("Andet")
                st.session_state["final_labels"] = new_final
                st.success("Kategorier opdateret ✔️")
                st.write(", ".join(new_final))
        with coly:
            st.info("Tip: Ret stavefejl, slå sammen eller tilføj nye. Disse labels bruges til selve klassifikationen.")

    # 3) Klassificér alle svar
    st.subheader("3) Klassificér alle svar (LLM)")
    batch_size = st.number_input("Batch-størrelse", min_value=10, max_value=150, value=30, step=10)  # lidt mindre batches → bedre fokus
    if st.session_state.get("final_labels") and st.button("Kør klassifikation på alle svar"):
        with st.spinner("Klassificerer alle svar..."):
            labels = st.session_state["final_labels"]
            assignments: List[List[str]] = []
            for i in range(0, len(texts_all), batch_size):
                batch = texts_all[i:i+batch_size]
                preds = llm_assign_batch(labels, batch, model_assign)
                assignments.extend(preds)
            # byg output
            cats1, cats2, cats3 = [], [], []
            for labs in assignments:
                labs = labs[:3] + ["","",""]
                cats1.append(labs[0]); cats2.append(labs[1]); cats3.append(labs[2])
            df_out = pd.DataFrame({"Svar": texts_all, "Kategori_1": cats1, "Kategori_2": cats2, "Kategori_3": cats3})
            st.session_state["df_out"] = df_out
            log_event("assigned_all", rows=len(df_out))

    df_out = st.session_state.get("df_out")
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
                pct = round((cnt/total*100.0), 1) if total else 0.0
                ex = df_data.loc[df_data["Kategori_1"] == cat, "Svar"].head(3).tolist() + ["",""]
                rows.append({"Kategori": cat, "Antal": cnt, "Andel %": pct,
                             "Citater 1": ex[0], "Citater 2": ex[1], "Citater 3": ex[2]})
            return pd.DataFrame(rows).sort_values("Antal", ascending=False)

        analysis_df = build_analysis(df_out, st.session_state.get("final_labels", []))
        st.dataframe(analysis_df, use_container_width=True)

        # Download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df_out.to_excel(w, sheet_name="Data", index=False)
            meta = pd.DataFrame([{
                "Dato": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "Antal svar": len(df_out),
                "Antal kategorier": len(st.session_state.get("final_labels", [])),
            }])
            meta.to_excel(w, sheet_name="Analyse", index=False, startrow=0)
            analysis_df.to_excel(w, sheet_name="Analyse", index=False, startrow=3)
        buf.seek(0)
        st.download_button("Download Excel (LLM-kategoriseret)",
                           data=buf, file_name="aabne_svar_LLM.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.divider()
st.subheader("Session-log")
download_audit_widget()
