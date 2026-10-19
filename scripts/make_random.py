import json, random, os
from huggingface_hub import hf_hub_download
home = os.path.expanduser("~")
p = hf_hub_download(repo_id="redsgnaoh/acereason11_100k", repo_type="dataset",
                    filename="data/acereason11_100k.json", local_dir=f"{home}/data_raw")
data = json.load(open(p))
print("extra_info example:", data[0].get("extra_info"))
random.seed(42)
subset = random.sample(data, 15000)
for x in subset:
    x.pop("extra_info", None)
out = f"{home}/LLaMA-Factory/data/random15k.json"
json.dump(subset, open(out, "w"), ensure_ascii=False)
info_path = f"{home}/LLaMA-Factory/data/dataset_info.json"
info = json.load(open(info_path))
info["random15k"] = {"file_name": "random15k.json"}
json.dump(info, open(info_path, "w"), indent=2, ensure_ascii=False)
print("Saved and registered", len(subset), "examples")
