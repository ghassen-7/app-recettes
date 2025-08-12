# app.py
# --------------------------------------------------------------
# Application Streamlit de recommandation de recettes par ingrédients
# MVP robuste + prêt pour extensions (TF‑IDF + Jaccard + filtres)
# --------------------------------------------------------------

import re
import ast
from typing import List, Set, Dict

import numpy as np
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ---- Optionnel : traducteur en→fr pour titres & instructions ----
try:
    from transformers import pipeline as hf_pipeline
    @st.cache_resource(show_spinner=False)
    def get_translator():
        return hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
except Exception:
    def get_translator():
        raise RuntimeError("Transformers indisponible. Installez: pip install transformers sentencepiece torch")

# -----------------------------
# ---------- Utils ------------
# -----------------------------

@st.cache_data(show_spinner=False)
def load_extra_synonyms(path: str = "synonyms_fr.csv") -> Dict[str, str]:
    """Charge un mapping optionnel FR: colonnes 'variant','canonical'. Non bloquant si absent."""
    try:
        df_syn = pd.read_csv(path)
    except Exception:
        return {}
    m: Dict[str, str] = {}
    for _, r in df_syn.iterrows():
        var = unidecode(str(r.get("variant", "")).lower().strip())
        can = unidecode(str(r.get("canonical", "")).lower().strip())
        if var and can:
            m[var] = can
    return m

UNITS = {
    "g", "gramme", "grammes", "grams", "kg", "kilogramme", "ml", "cl", "l", "litre", "litres",
    "tsp", "tbsp", "cuil", "cuillere", "cuillère", "cas", "cac", "c.a.s", "c.a.c",
    "pinch", "pincee", "pincée", "tranche", "tranches", "piece", "pièce", "pieces", "pièces",
    "boite", "boites", "boîte", "boîtes", "bouquet", "sachet", "sachets"
}

STOPWORDS = {
    "de", "des", "du", "la", "le", "les", "un", "une", "et", "ou", "à", "au", "aux", "d'",
    "en", "pour", "avec", "sans", "sur", "dans", "a", "the", "of"
}

# Quelques synonymes FR/EN courants pour matcher plus largement
SYNONYMS: Dict[str, str] = {
    # —— Canonicalisation en FR ——
    # légumes
    "zucchini": "courgette", "courgette": "courgette",
    "eggplant": "aubergine", "aubergine": "aubergine",
    "bellpepper": "poivron", "bell": "poivron", "poivron": "poivron",
    "corn": "mais", "maïs": "mais", "mais": "mais",
    "tomato": "tomate", "tomatoes": "tomate", "tomate": "tomate",
    "carrot": "carotte", "carrots": "carotte", "carotte": "carotte",
    "mushroom": "champignon", "mushrooms": "champignon", "champignon": "champignon",
    "spinach": "epinard", "épinard": "epinard", "epinard": "epinard",
    # aromatiques
    "garlic": "ail", "ail": "ail", "onion": "oignon", "onions": "oignon", "oignon": "oignon",
    "parsley": "persil", "persil": "persil", "cilantro": "coriandre", "coriandre": "coriandre",
    "basil": "basilic", "basilic": "basilic", "thyme": "thym", "thym": "thym",
    # protéines
    "beef": "boeuf", "bœuf": "boeuf", "boeuf": "boeuf",
    "chicken": "poulet", "poulet": "poulet",
    "pork": "porc", "porc": "porc",
    "turkey": "dinde", "dinde": "dinde",
    "tuna": "thon", "thon": "thon",
    "salmon": "saumon", "saumon": "saumon",
    "egg": "oeuf", "eggs": "oeuf", "oeuf": "oeuf",
    # légumineuses / céréales
    "peas": "petitspois", "petitspois": "petitspois",
    "chickpea": "pois chiche", "chickpeas": "pois chiche", "pois chiche": "pois chiche",
    "lentil": "lentille", "lentils": "lentille", "lentille": "lentille",
    "rice": "riz", "riz": "riz",
    "pasta": "pates", "noodles": "nouilles", "pâtes": "pates", "pates": "pates",
    # lait & annexes
    "milk": "lait", "lait": "lait", "cheese": "fromage", "fromage": "fromage",
    "yogurt": "yaourt", "yaourt": "yaourt", "butter": "beurre", "beurre": "beurre",
    "cream": "creme", "crème": "creme",
    # autres
    "lemon": "citron", "lime": "citron vert", "citron": "citron",
    "sugar": "sucre", "sucre": "sucre", "flour": "farine", "farine": "farine",
    "salt": "sel", "sel": "sel", "pepper": "poivre", "poivre": "poivre",
    "oil": "huile", "olive": "olive", "vinegar": "vinaigre", "vinaigre": "vinaigre",
    "coconut": "coco", "ginger": "gingembre", "curry": "curry"
}

COMMON_ING = [
    "poulet", "boeuf", "thon", "oeuf", "riz", "pates", "tomate", "oignon", "ail", "poivron",
    "courgette", "aubergine", "carotte", "fromage", "lait", "yaourt", "beurre", "farine", "sucre",
    "sel", "poivre", "huile", "citron", "persil", "coriandre", "mais"
]

@st.cache_data(show_spinner=False)
def load_data(path: str = "recipes.csv") -> pd.DataFrame:
    """Charge un CSV de recettes. Si absent, crée un mini jeu de données de secours.
    Colonnes attendues: id, title, ingredients, instructions, tags, minutes, url
    - ingredients: liste ou string séparée par '|'
    - tags: liste ou string séparée par '|'
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        data = [
            {
                "id": 1,
                "title": "Salade de thon express",
                "ingredients": "thon|tomate|oignon|mais|huile|sel|poivre",
                "instructions": "Mélanger tous les ingrédients, assaisonner et servir frais.",
                "tags": "rapide|frais|salade",
                "minutes": 10,
                "url": ""
            },
            {
                "id": 2,
                "title": "Pâtes à l'ail et au fromage",
                "ingredients": "pates|ail|fromage|beurre|sel|poivre",
                "instructions": "Cuire les pâtes. Faire fondre beurre+ail, mélanger avec fromage et pâtes.",
                "tags": "vegetarien|rapide",
                "minutes": 15,
                "url": ""
            },
            {
                "id": 3,
                "title": "Poulet aux poivrons",
                "ingredients": "poulet|poivron|oignon|ail|huile|sel|poivre",
                "instructions": "Saisir le poulet, ajouter légumes, mijoter 15 min.",
                "tags": "",
                "minutes": 30,
                "url": ""
            },
            {
                "id": 4,
                "title": "Curry de pois chiches",
                "ingredients": "pois chiche|tomate|oignon|ail|lait de coco|curry|sel",
                "instructions": "Revenir oignon+ail, ajouter tomate, curry, pois chiches, lait coco.",
                "tags": "vegan|vegetarien",
                "minutes": 25,
                "url": ""
            },
            {
                "id": 5,
                "title": "Ratatouille simplifiée",
                "ingredients": "courgette|aubergine|poivron|tomate|oignon|ail|huile|sel",
                "instructions": "Faire revenir oignon+ail, ajouter légumes coupés, mijoter 25 min.",
                "tags": "vegan",
                "minutes": 35,
                "url": ""
            },
        ]
        df = pd.DataFrame(data)

    # Harmoniser types
    def _to_list(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            if "[" in x and "]" in x:
                # liste sérialisée
                try:
                    arr = ast.literal_eval(x)
                    if isinstance(arr, list):
                        return arr
                except Exception:
                    pass
            return [t.strip() for t in re.split(r"[|,]", x) if t.strip()]
        return []

    df["ingredients"] = df["ingredients"].apply(_to_list)
    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(_to_list)
    else:
        df["tags"] = [[] for _ in range(len(df))]

    if "minutes" not in df.columns:
        df["minutes"] = None
    return df


def normalize_token(token: str) -> str:
    t = unidecode(token.lower())
    t = re.sub(r"[^a-zA-Z ]", " ", t)
    t = t.replace("  ", " ").strip()
    if t in STOPWORDS or t in UNITS or len(t) <= 1:
        return ""
    # mapping perso (CSV) puis mapping interne -> FR
    custom = load_extra_synonyms()
    t = custom.get(t, t)
    return SYNONYMS.get(t, t)


def tokenize_ingredient_line(line: str) -> List[str]:
    # supprime quantités et ponctuation
    s = unidecode(line.lower())
    s = re.sub(r"\d+[\,\.]?\d*", " ", s)  # nombres
    s = re.sub(r"[^a-zA-Z ]", " ", s)
    # explode
    raw_tokens = [t.strip() for t in s.split() if t.strip()]
    tokens = []
    for t in raw_tokens:
        if t in STOPWORDS or t in UNITS:
            continue
        t = normalize_token(t)
        if t:
            tokens.append(t)
    return tokens


def ingredients_to_tokens(ings: List[str]) -> List[str]:
    tokens: List[str] = []
    for ing in ings:
        tokens.extend(tokenize_ingredient_line(ing))
    # dédoublonnage léger mais on garde la fréquence pour TF‑IDF
    return tokens


@st.cache_data(show_spinner=False)
def build_index(df: pd.DataFrame):
    # Normalize + concat tokens par recette
    token_lists = [ingredients_to_tokens(ings) for ings in df["ingredients"]]
    df["_tokens"] = token_lists
    docs = [" ".join(toks) for toks in token_lists]

    tfidf = TfidfVectorizer(token_pattern=r"[^\s]+", lowercase=False)
    X = tfidf.fit_transform(docs)
    return df, tfidf, X


def score_recipes(user_tokens: List[str], df: pd.DataFrame, tfidf: TfidfVectorizer, X, 
                  required: Set[str], banned: Set[str], max_minutes: int | None,
                  alpha: float = 0.6, beta: float = 0.4) -> pd.DataFrame:
    # Similarité TF‑IDF
    user_doc = " ".join(user_tokens)
    if user_doc.strip():
        u_vec = tfidf.transform([user_doc])
        cos = cosine_similarity(u_vec, X)[0]
    else:
        cos = np.zeros(len(df))

    # Jaccard
    user_set = set(user_tokens)
    jacc = []
    for toks in df["_tokens"]:
        s = set(toks)
        if not user_set and not s:
            j = 0.0
        else:
            inter = len(user_set & s)
            union = len(user_set | s)
            j = inter / union if union else 0.0
        jacc.append(j)
    jacc = np.array(jacc)

    score = alpha * cos + beta * jacc

    # Pénalités/contraintes
    pen = np.zeros(len(df))

    # Must‑have: si un requis absent => grosse pénalité
    if required:
        req = set(normalize_token(r) for r in required)
        req.discard("")
        for i, toks in enumerate(df["_tokens"]):
            if not req.issubset(set(toks)):
                pen[i] += 0.5  # pénalité

    # Banned
    if banned:
        ban = set(normalize_token(b) for b in banned)
        ban.discard("")
        for i, toks in enumerate(df["_tokens"]):
            if set(toks) & ban:
                pen[i] += 0.7

    # Temps max
    if max_minutes is not None and "minutes" in df.columns:
        for i, m in enumerate(df["minutes"].fillna(9999)):
            if m > max_minutes:
                pen[i] += 0.3

    final = score - pen

    out = df.copy()
    out["score"] = final
    out["similarite_tfidf"] = cos
    out["similarite_jaccard"] = jacc

    # calcul des manquants/surplus vs input utilisateur pour affichage
    out["_missing"] = [sorted(list(set(user_set) - set(toks))) for toks in out["_tokens"]]
    out["_extras"] = [sorted(list(set(toks) - set(user_set))) for toks in out["_tokens"]]

    out = out.sort_values("score", ascending=False)
    return out


# -----------------------------
# --------- Streamlit ---------
# -----------------------------

st.set_page_config(page_title="Recettes par ingrédients", page_icon="🍳", layout="wide")

st.title("🍳 Recommandation de recettes par ingrédients")
st.caption("MVP: matching TF‑IDF + Jaccard, filtres et explications transparentes.")

with st.sidebar:
    st.header("Paramètres")
    translate_ui = st.checkbox("Traduire titres & instructions (en→fr)", value=False)
    st.caption("MarianMT local (Helsinki-NLP/opus-mt-en-fr).")
    st.write("Saisissez vos ingrédients (français ou anglais, séparés par des virgules).")
    default_text = "tomate, oignon, ail, pates, fromage"
    user_input = st.text_area("Ingrédients disponibles", value=default_text, height=90)

    st.write("Ou cliquez pour les ajouter rapidement :")
    cols = st.columns(3)
    chip_sel = []
    for i, ing in enumerate(COMMON_ING):
        if cols[i % 3].button(ing):
            chip_sel.append(ing)
    if chip_sel:
        user_input = (user_input + ", " + ", ".join(chip_sel)).strip(", ")

    req_input = st.text_input("Ingrédients OBLIGATOIRES (optionnel)")
    ban_input = st.text_input("Ingrédients INTERDITS (optionnel)")

    max_time = st.slider("Temps de préparation max (minutes)", min_value=5, max_value=120, value=45, step=5)

    topk = st.slider("Nombre de recettes à afficher", 3, 20, 8)

    alpha = st.slider("Poids TF‑IDF (alpha)", 0.0, 1.0, 0.6, 0.05)
    beta = 1.0 - alpha
    st.caption(f"Le poids Jaccard (beta) est automatiquement réglé à {beta:.2f}")

    st.markdown("---")
    st.download_button("Télécharger un CSV modèle", data=(
        "id,title,ingredients,instructions,tags,minutes,url\n"
        "101,Gratin de courgettes,\"courgette|fromage|lait|ail|sel|poivre\",\"Trancher, assembler, enfourner 25 min\",\"vegetarien\",30,\"\"\n"
    ), file_name="recipes_template.csv", mime="text/csv")

# Charger données et index
with st.spinner("Chargement des recettes…"):
    df = load_data()
    df, tfidf, X = build_index(df)

# Parse entrées utilisateur
raw_ings = [t.strip() for t in re.split(r"[,|]", user_input) if t.strip()]
user_tokens = []
for t in raw_ings:
    user_tokens.extend(tokenize_ingredient_line(t))

required = set([t.strip() for t in re.split(r"[,|]", req_input) if t.strip()])
banned = set([t.strip() for t in re.split(r"[,|]", ban_input) if t.strip()])

# Scoring
results = score_recipes(user_tokens, df, tfidf, X, required, banned, max_time)

# UI Résultats
left, right = st.columns([3, 2])
with left:
    st.subheader("Résultats")
    if len(user_tokens) == 0:
        st.info("Ajoutez au moins un ingrédient pour démarrer. Un mini jeu de données de démonstration est chargé si 'recipes.csv' est absent.")

    shown = results.head(topk)
    for _, row in shown.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['title']}")
            meta = []
            if pd.notna(row.get("minutes")):
                meta.append(f"⏱️ {int(row['minutes'])} min")
            if row.get("tags"):
                meta.append(" · ".join(row["tags"]))
            if meta:
                st.caption(" | ".join(meta))

            st.markdown(
                f"**Score**: {row['score']:.3f}  ·  TF‑IDF: {row['similarite_tfidf']:.3f}  ·  Jaccard: {row['similarite_jaccard']:.3f}"
            )

            # Diff d'ingrédients
            miss = [m for m in row["_missing"] if m]
            extra = [e for e in row["_extras"] if e]
            if miss:
                st.warning("Manquants: " + ", ".join(miss))
            if extra:
                st.success("En plus dans la recette: " + ", ".join(extra))

            with st.expander("Voir les ingrédients & instructions"):                # Titre/Instructions (option de traduction en→fr)
                title_text = str(row["title"]) if pd.notna(row["title"]) else ""
                instr_text = str(row["instructions"]) if pd.notna(row["instructions"]) else ""
                if translate_ui:
                    try:
                        tr = get_translator()
                        if title_text:
                            title_text = tr(title_text)[0]["translation_text"]
                        if instr_text:
                            instr_text = tr(instr_text)[0]["translation_text"]
                    except Exception:
                        st.info("Traduction indisponible (vérifiez les dépendances).")
                st.markdown("**Titre** : " + title_text)
                st.markdown("**Ingrédients (normalisés FR)** : " + ", ".join(sorted(set(row["_tokens"]))))
                st.markdown("**Instructions** : " + instr_text)
                if row.get("url"):
                    st.markdown(f"[Source]({row['url']})")

with right:
    st.subheader("Explications du matching")
    st.write(
        """
        Le score combine deux signaux :
        - **TF‑IDF (alpha)** : similarité cosinus entre vos ingrédients et ceux de la recette, pondérant les ingrédients rares.
        - **Jaccard (beta)** : chevauchement ensembliste entre vos ingrédients et ceux de la recette.
        
        Des pénalités s'appliquent si:
        - un ingrédient *obligatoire* manque
        - un ingrédient *interdit* est présent
        - le temps dépasse la limite choisie
        """
    )

    st.markdown("**Ingrédients interprétés**")
    if user_tokens:
        st.code(", ".join(user_tokens))
    else:
        st.code("(aucun)")

    st.markdown("**Astuces**")
    st.write(
        """
        - Ajoutez vos propres recettes via `recipes.csv` (même dossier) – utilisez le modèle téléchargeable.
        - Utilisez les champs *OBLIGATOIRES* / *INTERDITS* pour affiner.
        - Réglez le curseur **alpha** pour donner plus (ou moins) d'importance aux ingrédients rares.
        """
    )

st.markdown("---")
st.caption("Fichier requis optionnel: `recipes.csv` avec les colonnes: id,title,ingredients,instructions,tags,minutes,url. La colonne `ingredients` utilise '|' pour séparer les items.")
