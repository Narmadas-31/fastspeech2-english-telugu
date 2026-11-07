import os, yaml
CFG = "/content/project/Telugu/config.yaml"
ENG = "/content/project/English/config.yaml"
D = "/content/drive/MyDrive/fs2/output/Telugu"
def load(p):
    return yaml.safe_load(open(p, encoding="utf8")) if os.path.exists(p) else None
cfg = load(CFG)
if not cfg:
    cfg = load(ENG)
    assert cfg, "Telugu and English config.yaml are both empty"
    print("Telugu config.yaml was empty -> using the English config as the base")
pp = cfg["preprocessing"]
pp["dataset"] = "Telugu"
pp.setdefault("path", {}).update(
    corpus_path="/content/raw_data/Telugu",
    raw_path="/content/raw_data/Telugu",
    preprocessed_path="/content/preprocessed_data/Telugu",
    lexicon_path="/content/drive/MyDrive/fs2/backup/telugu_dict.txt",
)
pp.pop("text", None)
txt = pp["preprocessing"]["text"]
txt["language"] = "te"
txt["text_cleaners"] = ["basic_cleaners"]
if "multi_speaker" in cfg["model"]:
    cfg["model"]["multi_speaker"] = True
cfg["train"]["path"] = {"ckpt_path": f"{D}/ckpt", "log_path": f"{D}/log", "result_path": f"{D}/result"}
cfg["train"]["step"]["total_step"] = 10000
cfg["train"]["step"]["save_step"] = 1000
yaml.safe_dump(cfg, open(CFG, "w", encoding="utf8"), sort_keys=False, allow_unicode=True)
print("config patched: language =", txt["language"], "| cleaners =", txt["text_cleaners"])
