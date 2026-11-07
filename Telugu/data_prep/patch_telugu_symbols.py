import glob, json, os
import tgt
T = "/content/project/Telugu"
DRIVE_JSON = "/content/drive/MyDrive/fs2/telugu_phones.json"
files = glob.glob("/content/preprocessed_data/Telugu/TextGrid/*/*.TextGrid")
if files:
    ph = set()
    for f in files:
        tier = tgt.io.read_textgrid(f).get_tier_by_name("phones")
        for it in (getattr(tier, "intervals", None) or tier._objects):
            if it.text.strip():
                ph.add(it.text.strip())
    ph -= {"sil", "sp", "spn"}
    phones = sorted(ph)
    json.dump(phones, open(DRIVE_JSON, "w", encoding="utf8"), ensure_ascii=False)
else:
    phones = json.load(open(DRIVE_JSON, encoding="utf8"))
json.dump(phones, open(f"{T}/text/telugu_phones.json", "w", encoding="utf8"), ensure_ascii=False)
path = f"{T}/text/symbols.py"
sy = open(path, encoding="utf8").read()
if "TELUGU_PHONES" not in sy:
    sy += """

# --- TELUGU_PHONES: Telugu grapheme symbols produced by the MFA alignment ---
import json as _json, os as _os
try:
    _tp = _json.load(open(_os.path.join(_os.path.dirname(__file__), "telugu_phones.json"), encoding="utf8"))
except Exception:
    _tp = []
symbols = symbols + [s for s in ["@" + p for p in _tp] if s not in symbols]
"""
    open(path, "w", encoding="utf8").write(sy)
print(len(phones), "Telugu symbols")
