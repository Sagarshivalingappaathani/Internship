import json
import random
import pandas as pd
from itertools import combinations

# === Load the IFIXIT dataset (line-delimited JSON) ===
with open("Appliance.json", "r") as f:
    documents = [json.loads(line) for line in f if line.strip()]

# === Extract valid guides with subject and toolbox ===
guides = []
docid_to_title = {}

for idx, doc in enumerate(documents):
    subject = doc.get("Subject", "").strip().lower()
    toolbox = doc.get("Toolbox", [])
    tags = set(
        tag.get("Name", "").strip().lower()
        for tag in toolbox if isinstance(tag, dict) and tag.get("Name")
    )
    if subject and tags:
        guides.append({
            "original_id": idx,
            "title": doc.get("Title", ""),
            "subject": subject,
            "tags": tags
        })
        docid_to_title[idx] = doc.get("Title", "")

# === Group by subject to prepare positive pairs ===
subject_groups = {}
for g in guides:
    subject_groups.setdefault(g["subject"], []).append(g)

# === Generate positive pairs: same subject + ≥2 overlapping tags ===
positive_pairs = []
for group in subject_groups.values():
    for g1, g2 in combinations(group, 2):
        if len(g1["tags"].intersection(g2["tags"])) >= 2:
            positive_pairs.append((g1["original_id"], g2["original_id"], 1))

# === Generate negative pairs: different subjects ===
negative_pairs = []
while len(negative_pairs) < len(positive_pairs):
    g1, g2 = random.sample(guides, 2)
    if g1["subject"] != g2["subject"]:
        negative_pairs.append((g1["original_id"], g2["original_id"], 0))

# === Combine and shuffle all pairs ===
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)

# === Save as DataFrame ===
df = pd.DataFrame(all_pairs, columns=["doc1_id", "doc2_id", "label"])
df["doc1_title"] = df["doc1_id"].apply(lambda i: docid_to_title.get(i, ""))
df["doc2_title"] = df["doc2_id"].apply(lambda i: docid_to_title.get(i, ""))

# === Export to CSV ===
df.to_csv("labeled_pairs_rule_based.csv", index=False)
print(f"Saved {len(df)} labeled pairs to labeled_pairs_rule_based.csv")
