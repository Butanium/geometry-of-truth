from datasets import load_dataset
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
ds = load_dataset("OpenAssistant/oasst1")
print(ds["train"][0]["text"], ds["train"][0]["role"], ds["train"][0]["lang"])

dataset = {"statement": [], "is_en": []}

en_counter = 0
es_counter = 0

for entry in ds["train"]:
    if entry["role"] == "prompter" and entry["lang"] in ["en", "es"]:
        if entry["lang"] == "en" and en_counter > 7500:
            continue
        dataset["statement"].append(entry["text"])
        dataset["is_en"].append(entry["lang"] == "en")
        if entry["lang"] == "en":
            en_counter += 1
        else:
            es_counter += 1

print(en_counter, es_counter)

df = pd.DataFrame.from_dict(dataset)
df.to_csv(ROOT / "assistant_prompts.csv")
