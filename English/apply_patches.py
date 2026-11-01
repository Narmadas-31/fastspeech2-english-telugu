import pathlib, re

def edit(path, fn):
    p = pathlib.Path(path)
    s = p.read_text(encoding="utf8")
    new = fn(s)
    if new != s:
        p.write_text(new, encoding="utf8")
        print("patched:", path)

# 1) preprocessor: MFA 2/3 writes silences as "" -> map to "sp"; split train/val by filename prefix
def fix_preprocessor(s):
    if 'p = "sp"' not in s:
        s, n = re.subn(
            r"(\n([ \t]*)s, e, p = t\.start_time, t\.end_time, t\.text\n)",
            lambda m: m.group(1) + f'{m.group(2)}if p == "":\n{m.group(2)}    p = "sp"\n', s)
        assert n == 1, "get_alignment line not found"
    if "train_lines" not in s:
        s, n = re.subn(
            r'\n([ \t]*)(with open\(os\.path\.join\(self\.out_dir, "train\.txt"\))',
            lambda m: ("\n{i}train_lines = [m for m in out if m.split('|')[0].startswith('train_')]"
                       "\n{i}val_lines = [m for m in out if m.split('|')[0].startswith('val_')]"
                       "\n{i}print('train utts:', len(train_lines), 'val utts:', len(val_lines))"
                       "\n{i}{w}").format(i=m.group(1), w=m.group(2)), s)
        assert n == 1, "train.txt writer not found"
        s = re.sub(r"out\[\s*self\.val_size\s*:\s*\]", "train_lines", s)
        s = re.sub(r"out\[\s*:\s*self\.val_size\s*\]", "val_lines", s)
    return s
edit("preprocessor/preprocessor.py", fix_preprocessor)

# 2) newer librosa needs keyword arguments
def fix_librosa(s):
    s = re.sub(r"librosa_mel_fn\(\s*sampling_rate,\s*filter_length,\s*n_mel_channels,\s*mel_fmin,\s*mel_fmax\s*\)",
               "librosa_mel_fn(sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)", s)
    return re.sub(r"pad_center\((\w+),\s*(\w+)\)", r"pad_center(\1, size=\2)", s)
for f in pathlib.Path("audio").glob("*.py"):
    edit(f, fix_librosa)

# 3) removed numpy aliases (np.int, np.float, np.bool)
for f in pathlib.Path(".").rglob("*.py"):
    if f.name != "apply_patches.py":
        edit(f, lambda s: re.sub(r"\bnp\.(int|float|bool)\b", r"\1", s))

# 4) train.py: read the single config.yaml
NEW_MAIN = '''if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    configs = (cfg["preprocessing"], cfg["model"], cfg["train"])
    main(args, configs)
'''
def fix_train(s):
    if 'cfg["preprocessing"]' in s:
        return s
    return s.split('if __name__ == "__main__":')[0] + NEW_MAIN
edit("train.py", fix_train)

# 5) preprocess.py: read the single config.yaml
pathlib.Path("preprocess.py").write_text('''import argparse
import yaml
from preprocessor.preprocessor import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="config.yaml", help="path to config.yaml")
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    Preprocessor(cfg["preprocessing"]).build_from_path()
''')
print("patched: preprocess.py")