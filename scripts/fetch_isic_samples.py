import json
import urllib.request

queries = {
    "Melanocytic Nevi (nv)": "diagnosis_3=Nevus",
    "Melanoma (mel)": "diagnosis_3=Melanoma%2C%20NOS",
    "Basal Cell Carcinoma (bcc)": "diagnosis_3=Basal%20cell%20carcinoma",
    "Actinic Keratosis (akiec)": "diagnosis_3=Actinic%20keratosis",
    "Benign Keratosis (bkl)": "diagnosis_3=Seborrheic%20keratosis",
    "Dermatofibroma (df)": "diagnosis_3=Dermatofibroma",
    "Vascular Lesions (vasc)": "diagnosis_3=Hemangioma",
}

for label, q in queries.items():
    url = f"https://api.isic-archive.com/api/v2/images?limit=1&{q}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.load(r)
        if data["results"]:
            img = data["results"][0]
            print(label)
            print(f"  page:  https://isic-archive.com/search/images/{img['isic_id']}")
            print(f"  image: {img['files']['full']['url']}")
            print(f"  dx:    {img['metadata']['clinical'].get('diagnosis_3', '?')}")
        else:
            print(f"{label} - no results")
    except Exception as e:
        print(f"{label} - error: {e}")
    print()
