import yaml
BASE = "config_ref/LibriTTS"
load = lambda n: yaml.load(open(f"{BASE}/{n}.yaml"), Loader=yaml.FullLoader)
pre, model, train = load("preprocess"), load("model"), load("train")

pre["dataset"] = "Svarah"
pre["path"].update(
    corpus_path="/content/drive/MyDrive/fs2/raw_all/svarah",
    lexicon_path="/content/drive/MyDrive/fs2/svarah_extended.dict",
    raw_path="/content/raw_data/Svarah",
    preprocessed_path="/content/preprocessed_data/Svarah",
)
pre["preprocessing"]["val_size"] = 0            # split is by filename prefix (see apply_patches.py)
pre["preprocessing"]["text"]["text_cleaners"] = ["english_cleaners"]

model["multi_speaker"] = True
model["max_seq_len"] = 1500                      # 15 s of audio = ~1300 mel frames
model["vocoder"] = {"model": "HiFi-GAN", "speaker": "universal"}

out = "/content/drive/MyDrive/fs2/output/English"
train["path"] = {"ckpt_path": f"{out}/ckpt", "log_path": f"{out}/log", "result_path": f"{out}/result"}
train["optimizer"]["batch_size"] = 16
train["optimizer"]["anneal_steps"] = [20000, 26000]
train["step"].update(total_step=30000, log_step=100, synth_step=1000, val_step=1000, save_step=2000)

cfg = {"preprocessing": pre, "model": model, "train": train}
yaml.safe_dump(cfg, open("config.yaml", "w"), sort_keys=False)
print("wrote config.yaml")