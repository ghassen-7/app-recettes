# prepare_foodcom.py
import pandas as pd, ast, os, sys

src = os.path.join("data", "RAW_recipes.csv")
if not os.path.exists(src):
    sys.exit(f"Introuvable: {src}")

def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            return v if isinstance(v, list) else [x]
        except Exception:
            return [x]
    return []

def join_list(x):
    return "|".join(str(t).strip() for t in to_list(x) if str(t).strip())

df = pd.read_csv(src)

out = pd.DataFrame({
    "id": df["id"],
    "title": df["name"],
    "ingredients": df["ingredients"].apply(join_list),
    "instructions": df["steps"].apply(lambda s: " ".join(to_list(s))),
    "tags": df["tags"].apply(join_list),
    "minutes": df["minutes"],
    "url": ""
})

# Option: filtrer pour rester raisonnable en local
if "n_ingredients" in df.columns:
    mask = (df["minutes"] <= 120) & (df["n_ingredients"] <= 25)
    out = out[mask]

# Option: échantillon si RAM limitée
# out = out.sample(50000, random_state=42)

out.to_csv("recipes.csv", index=False)
print("OK -> recipes.csv", len(out))
