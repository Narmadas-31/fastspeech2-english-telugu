import argparse
import csv
import os
import re
from string import punctuation

import numpy as np
import torch
import yaml
import json
import inflect
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from g2p_en import G2p
from scipy.io import wavfile

from utils.model import get_model, get_vocoder, vocoder_infer
from text import text_to_sequence
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_SENTENCES = [
    ("E1", "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."),
    ("E2a", "The orchestra's conduct during the ceremony was praised by the audience."),
    ("E2b", "The researchers will conduct an experiment to evaluate the proposed method."),
    ("E3", "The rhyth m of the orchestra changed unexpectedly."),
    ("E4", "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."),
]

_inf = inflect.engine()


def _words(n):
    return re.sub(r"[,\-]", " ", _inf.number_to_words(int(n)))


def _say_year(n):
    if 2000 <= n <= 2009:
        return _words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return _words(hi) + " hundred"
    if lo < 10:
        return _words(hi) + " oh " + _words(lo)
    return _words(hi) + " " + _words(lo)


def _say_time(m):
    h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3).lower()
    out = _words(h)
    if mi == 0:
        out += " o clock"
    elif mi < 10:
        out += " oh " + _words(mi)
    else:
        out += " " + _words(mi)
    return out + (" __am__" if ap == "a" else " __pm__")


def normalize_text(text):
    text = re.sub(r"(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?", _say_time, text)

    def num(m):
        n = int(m.group(0))
        return _say_year(n) if 1100 <= n <= 2099 else _words(n)

    return re.sub(r"\d+", num, text)


SPECIAL_PHONES = {
    "__am__": ["EY1", "EH1", "M"],
    "__pm__": ["P", "IY1", "EH1", "M"],
}

HETERONYMS = {
    "conduct": {
        "noun": ["K", "AA1", "N", "D", "AH0", "K", "T"],
        "verb": ["K", "AH0", "N", "D", "AH1", "K", "T"],
    }
}
VERB_TRIGGERS = {"will", "to", "can", "could", "would", "should", "must", "may",
                 "might", "shall", "we", "they", "i", "you", "not"}


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            t = line.strip().split()
            if len(t) < 2:
                continue
            word, phones = t[0].lower(), t[1:]
            while phones and re.fullmatch(r"[0-9.eE+-]+", phones[0]):
                phones = phones[1:]
            if phones and word not in lexicon:
                lexicon[word] = phones
    return lexicon


def text_to_phones(text, lexicon, g2p, fix_heteronyms=False):
    text = normalize_text(text).rstrip(punctuation)
    phones, prev = [], ""
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        if w == "" or w.isspace():
            continue
        lw = w.lower()
        if lw in SPECIAL_PHONES:
            phones += SPECIAL_PHONES[lw]
        elif fix_heteronyms and lw in HETERONYMS:
            kind = "verb" if prev in VERB_TRIGGERS else "noun"
            phones += HETERONYMS[lw][kind]
        elif lw in lexicon:
            phones += lexicon[lw]
        else:
            phones += [p for p in g2p(w) if p != " "]
        if re.match(r"\w", lw):
            prev = lw
    return ["sp" if not re.match(r"\w", p) else p for p in phones]


def _fig_width(n):
    return min(28, max(9, 0.33 * n))


def plot_duration(durations_ms, labels, title, path):
    fig, ax = plt.subplots(figsize=(_fig_width(len(labels)), 4))
    ax.bar(range(len(labels)), durations_ms, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Predicted duration (ms)")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curve(values, durations, labels, ylabel, title, path, color, hop, sr, frame_level):
    durations = durations.astype(int)
    frames = values if frame_level else np.repeat(values, durations)
    t = np.arange(len(frames)) * hop / sr
    ends = np.cumsum(durations) * hop / sr
    mids = ends - durations * hop / sr / 2
    fig, ax = plt.subplots(figsize=(_fig_width(len(labels)), 4))
    ax.plot(t, frames, color=color, lw=1.6)
    for e in ends:
        ax.axvline(e, color="gray", lw=0.3, ls="--")
    ax.set_xticks(mids)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlim(0, t[-1] if len(t) else 1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s), phoneme labels at centres")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_one(sid, text, model, vocoder, lexicon, g2p, configs, stats, args, dirs):
    pre, mcfg, tcfg = configs
    pp = pre["preprocessing"]
    sr, hop = pp["audio"]["sampling_rate"], pp["stft"]["hop_length"]

    phones = text_to_phones(text, lexicon, g2p, args.fix_heteronyms)
    print("\n[{}] {}".format(sid, text))
    print("    phonemes:", " ".join(phones))
    seq = np.array(
        text_to_sequence("{" + " ".join(phones) + "}", pp["text"]["text_cleaners"])
    )
    labels = [symbols[i].lstrip("@") for i in seq]

    speakers = torch.tensor([args.speaker_id]).long().to(device)
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(device)
    src_lens = torch.tensor([len(seq)]).to(device)

    with torch.no_grad():
        out = model(
            speakers, texts, src_lens, len(seq),
            p_control=args.pitch_control,
            e_control=args.energy_control,
            d_control=args.duration_control,
        )
        mel_len = out[9][0].item()
        wav = vocoder_infer(
            out[1].transpose(1, 2), vocoder, mcfg, pre, lengths=out[9] * hop
        )[0]

    wavfile.write(os.path.join(dirs["audio"], sid + ".wav"), sr, wav)

    dur = out[5][0, : len(seq)].cpu().numpy()
    pitch = out[2][0].cpu().numpy()
    energy = out[3][0].cpu().numpy()
    p_frame = pp["pitch"]["feature"] != "phoneme_level"
    e_frame = pp["energy"]["feature"] != "phoneme_level"
    if not p_frame:
        pitch = pitch[: len(seq)]
    if not e_frame:
        energy = energy[: len(seq)]

    if pp["pitch"].get("normalization", True):
        pitch = pitch * stats["pitch"][3] + stats["pitch"][2]
    if pp["energy"].get("normalization", True):
        energy = energy * stats["energy"][3] + stats["energy"][2]
    pitch = np.clip(pitch, 0, None)

    dur_ms = dur * hop / sr * 1000.0
    plot_duration(dur_ms, labels, "{} - predicted duration".format(sid),
                  os.path.join(dirs["duration"], sid + ".png"))
    plot_curve(pitch, dur, labels, "Predicted F0 (Hz)", "{} - predicted pitch".format(sid),
               os.path.join(dirs["pitch"], sid + ".png"), "tomato", hop, sr, p_frame)
    plot_curve(energy, dur, labels, "Predicted energy", "{} - predicted energy".format(sid),
               os.path.join(dirs["energy"], sid + ".png"), "darkviolet", hop, sr, e_frame)

    if not p_frame and not e_frame:
        with open(os.path.join(dirs["values"], sid + ".csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "phoneme", "duration_frames", "duration_ms", "pitch_hz", "energy"])
            for i, l in enumerate(labels):
                w.writerow([i, l, int(dur[i]), round(float(dur_ms[i]), 2),
                            round(float(pitch[i]), 2), round(float(energy[i]), 4)])
    print("    audio: {:.2f} s, {} phonemes".format(mel_len * hop / sr, len(seq)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, nargs="?", default="config.yaml")
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--lexicon", type=str, default=None)
    parser.add_argument("--out", type=str, default="../Results/English")
    parser.add_argument("--speaker_id", type=int, default=0)
    parser.add_argument("--fix_heteronyms", action="store_true")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--id", type=str, default="custom")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    configs = (cfg["preprocessing"], cfg["model"], cfg["train"])
    pre, mcfg, tcfg = configs
    if args.ckpt_dir:
        tcfg["path"]["ckpt_path"] = args.ckpt_dir

    dirs = {
        "audio": os.path.join(args.out, "test_generated_audios"),
        "duration": os.path.join(args.out, "variance_adaptor_plots", "duration"),
        "pitch": os.path.join(args.out, "variance_adaptor_plots", "pitch"),
        "energy": os.path.join(args.out, "variance_adaptor_plots", "energy"),
        "values": os.path.join(args.out, "variance_adaptor_plots", "values"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(mcfg, device)
    lexicon = read_lexicon(args.lexicon or pre["path"]["lexicon_path"])
    g2p = G2p()

    sentences = [(args.id, args.text)] if args.text else TEST_SENTENCES
    for sid, text in sentences:
        run_one(sid, text, model, vocoder, lexicon, g2p, configs, stats, args, dirs)
    print("\nDone. Results saved in", args.out)