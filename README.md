# FastSpeech 2 Text-to-Speech: English and Telugu

Implementation of **FastSpeech 2** for English and Telugu, with a separate model and config for each language.
The code is adapted from [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), following the paper
[FastSpeech 2 (Ren et al., ICLR 2021)](https://arxiv.org/abs/2006.04558). Waveforms come from the HiFi-GAN universal vocoder.

| Language | Dataset | Status |
|---|---|---|
| English | **Svarah** (AI4Bharat, Indian-accented English) | Complete: preprocessed, trained, inference run |
| Telugu | **IndicTTS Telugu**, IITM Speech Lab / IIT Madras (`SPRINGLab/IndicTTS_Telugu` on Hugging Face) | **Partially complete**: data preparation, alignment and preprocessing done; training not finished; no results (see Section 2) |

Dataset credit: the Telugu data is the IndicTTS database from the IITM Speech Lab (IIT Madras), used under the *License For Use of Indic TTS*. Svarah is from AI4Bharat.

---

## 1. Repository structure

```
project/
├── English/
│   ├── train.py  preprocess.py  inference.py  config.yaml
│   ├── dataset.py  plot_losses.py  evaluate.py  synthesize.py
│   ├── model/  transformer/  audio/  text/  preprocessor/  utils/  hifigan/
│   └── checkpoints/
├── Telugu/
│   ├── train.py  preprocess.py  inference.py  config.yaml
│   ├── dataset.py  evaluate.py  synthesize.py
│   ├── model/  transformer/  audio/  text/  preprocessor/  utils/  hifigan/
│   └── checkpoints/            (empty: see Section 2)
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/{duration,pitch,energy}/
│   └── Telugu/                 (empty: see Section 2)
├── Report/Technical_Report.pdf
├── README.md
└── requirements.txt
```

---

## 2. Submission status (what is and is not included)

| Requirement | English | Telugu |
|---|---|---|
| Preprocessing, training and inference code | Included and executed | Included. Preprocessing executed; training started; inference **not executed** |
| Modified reference files + separate `config.yaml` | Included | Included |
| Trained checkpoint | Included (`English/checkpoints/`, step 10000) | **Not included**: Telugu training was not completed |
| Training / validation loss plots | Included (`Results/English/`) | **Not included**: not generated |
| Generated audio for all 5 test sentences (E1, E2a, E2b, E3, E4) | Included | **Not included**: inference not run |
| Variance-adaptor plots (duration, pitch, energy) per test sentence | Included | **Not included**: inference not run |
| Required folder structure | Yes | Yes (Telugu result folders are empty) |

### Telugu: what was actually run

The Telugu notebook was executed up to **section 29**. Sections 30 onward (loss plots, inference with different checkpoints, listening to audio, variance-adaptor plots, collecting final results) were **not run**.

| Notebook section | Step | Done |
|---|---|---|
| 1–7 | Colab setup, load IndicTTS Telugu, dataset statistics | Yes |
| 8–11 | Text cleaning, 5 h train / 1 h val selection, overlap check, backup | Yes |
| 12–16 | Montreal Forced Aligner install, grapheme dictionary, aligner trained on our data, TextGrids | Yes (2510 of 2533) |
| 17–24 | Repo check, config and symbol patches, feature extraction, split verification | Yes |
| 25–27 | Training setup, 200-step smoke test | Yes |
| 28–29 | Full training launched (10,000-step plan), resume after disconnect | Started; **not completed, no checkpoint submitted** |
| 30 | Plot training / validation loss | **Not run** |
| 31–33 | Inference on T1, T2a, T2b, T3, T4; listen; variance-adaptor plots | **Not run** |
| 34–35 | Collect final results and checkpoint | **Not run** |

Consequently, the Telugu `inference.py` has not been tested. The Telugu test sentences (T1–T4), Telugu number-to-word handling for T4, and `plot_losses.py` for Telugu are not verified. Telugu results and analysis are not reported.

---

## 3. Dataset preparation

Both languages: about 5 h train and 1 h validation, 22.05 kHz mono, peak-normalised, one `.wav` and one `.lab` transcript per clip, aligned with Montreal Forced Aligner (MFA). Random selection uses `seed = 42`.

### English: Svarah (completed)
- Loaded with `load_dataset("ai4bharat/Svarah")`; all splits pooled (about 9.6 h).
- **Filtering:** clips between 2 s and 15 s; drop transcripts containing long digit strings (6 or more digits) and the words million, billion, trillion or quadrillion.
- **Text cleaning:** remove bracketed tags, expand numbers and ordinals with `inflect`, lowercase, keep only `a–z`, apostrophe and space.
- **Speakers:** 8 gender × age-group labels (`Female-18-30`, `Female-30-45`, `Female-45-60`, `Female-60`, `Male-18-30`, `Male-30-45`, `Male-45-60`, `Male-60`). These are group labels, not individual speakers.
- **Selection:** seeded shuffle; 5.00 h train (3056 clips) and 1.00 h validation (613 clips). No clip overlap and no transcript overlap (checked).
- **Alignment:** MFA with pretrained `english_us_arpa`. Out-of-vocabulary words (1160 types) were given pronunciations by the MFA G2P model and appended to an extended dictionary.
- **After alignment and feature extraction:** 3047 train and 613 val utterances (about 3.89 h and 0.79 h).

### Telugu: IndicTTS Telugu, IITM Speech Lab (data preparation completed)
- Source: 8576 utterances (about 36.7 h), one female and one male speaker, 48 kHz.
- **Text cleaning:** NFC normalisation, remove zero-width joiners, keep only Telugu characters (`U+0C00–U+0C7F`) and spaces. Sentences with digits or Latin letters were excluded from training data.
- **Filtering:** clips between 2 s and 15 s, leaving 5205 usable clips (about 11.8 h).
- **Selection:** time budget split equally between the two speakers (2.5 h + 2.5 h train, 0.5 h + 0.5 h val); 5.00 h train (2110 clips) and 1.00 h val (423 clips). Audio overlap and sentence overlap were checked and are both empty.
- **Alignment:** no ready Telugu dictionary was used. Each Telugu **character is one "phone"** (`కాలం → క ా ల ం`) and an MFA acoustic model was trained from scratch on the selected data (monophone + triphone). 2510 of 2533 TextGrids were produced (20 train and 3 val clips unaligned).
- **After alignment and feature extraction:** 2090 train and 420 val utterances (about 3.44 h and 0.69 h). The 63 grapheme symbols found in the alignments are stored in `Telugu/text/telugu_phones.json` and added to `text/symbols.py`.
- **Not included in this repo:** the Telugu audio, the prepared dataset, TextGrids and extracted features.

---

## 4. Modifications to the reference implementation

**Both languages**
- One `config.yaml` per language (`preprocessing`, `model`, `train` sections) in place of the original `config/<dataset>/*.yaml` files; `dataset.py` and `preprocess.py` read it.
- Compatibility fixes for current library versions:
  - `tier._objects` → `tier.intervals` (`tgt`).
  - `torch.load(..., map_location=device, weights_only=False)`.
  - `model.requires_grad_(False)`.
  - Keyword arguments for `librosa` mel filters and `pad_center`.
  - Pitch and energy cast to `float32`.
  - Import path `utils.tools.to_device`.
- `inference.py` synthesises the test sentences, saves one `.wav` per sentence, and saves predicted **duration, pitch (F0) and energy** plots per sentence. Flags: `--restore_step`, `--speaker_id`, `--out`.

**English only**
- `plot_losses.py` creates training and validation loss plots from the training logs.
- `g2p_en` handles out-of-vocabulary words (e.g. *antidisestablishmentarianism*).
- Optional `--fix_heteronyms` flag for test sentence E2 (*conduct* as noun vs. verb); off by default.

**Telugu only (implemented, not validated end to end)**
- Telugu text cleaning, grapheme symbol set (`@`-prefixed), and a replaced `preprocess.py` that reads `config.yaml` and goes straight to feature extraction from the wavs and TextGrids.
- Telugu paths and `total_step: 10000` set in `Telugu/config.yaml`.
- `Telugu/hifigan/generator_universal.pth.tar` is included in the repo for convenience.

---

## 5. Training configuration

| | English | Telugu |
|---|---|---|
| Training audio selected | 5 h (3056 clips) | 5 h (2110 clips) |
| Validation audio selected | 1 h (613 clips) | 1 h (423 clips) |
| Sampling rate | 22.05 kHz | 22.05 kHz |
| Speakers | 8 gender/age groups | 2 (Female, Male) |
| Total steps | 10,000 | 10,000 planned; **not completed** |
| Vocoder | HiFi-GAN universal | HiFi-GAN universal |
| Batch size / optimizer | see `English/config.yaml` | see `Telugu/config.yaml` |
| Hardware | Google Colab GPU | Google Colab GPU |

English training was resumed from checkpoints after Colab disconnects. Checkpoints at steps 2000, 6000 and 10000 were compared and **step 10000** is used for the reported English results. English inference used `--speaker_id 2` (`Female-30-45`).

---

## 6. Reproducibility

### Environment
```bash
git clone https://github.com/Narmadas-31/fastspeech2-english-telugu.git project
cd project
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner   # MFA is not on pip
```
Download the HiFi-GAN **universal** generator (`generator_universal.pth.tar`) from the [ming024/FastSpeech2 repo](https://github.com/ming024/FastSpeech2/tree/master/hifigan) into `English/hifigan/` (and `Telugu/hifigan/` if it is missing).

### English
```bash
cd English
# 1. Prepare wav + lab from ai4bharat/Svarah (needs a Hugging Face token); selection as in Section 3

# 2. Alignment
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
mfa g2p oov_words.txt english_us_arpa oov_pron.dict
cat oov_pron.dict >> svarah_extended.dict
mfa align --clean -j 2 raw_data/Svarah svarah_extended.dict english_us_arpa preprocessed_data/Svarah/TextGrid

# 3. Features (mel, pitch, energy, duration)
python preprocess.py config.yaml

# 4. Train
python train.py config.yaml                       # resume: --restore_step 6000

# 5. Loss plots
python plot_losses.py config.yaml ../Results/English

# 6. Inference and variance-adaptor plots
python inference.py config.yaml --restore_step 10000 --speaker_id 2 --out ../Results/English
```

### Telugu (steps 1–4 were run; steps 5–7 were not)
```bash
cd Telugu
# 1. Prepare wav + lab from SPRINGLab/IndicTTS_Telugu (IITM IndicTTS); selection as in Section 3

# 2. Grapheme dictionary (word<TAB>c h a r s) and aligner trained from scratch
mfa train --clean -j 2 --config_path mfa_config.yaml \
  --output_directory telugu_textgrid raw_data/Telugu telugu_dict.txt telugu_acoustic.zip

# 3. Features
python preprocess.py config.yaml

# 4. Train (launched, not completed)
python train.py config.yaml                       # resume: --restore_step <N>

# 5-7. NOT RUN: loss plots, inference, variance-adaptor plots
```
`mfa_config.yaml` used a shortened training plan: monophone 20 iterations and triphone 20 iterations (2000 leaves).

### Checkpoints
| Model | File | Status |
|---|---|---|
| English | `English/checkpoints/10000.pth.tar` | `https://github.com/Narmadas-31/fastspeech2-english-telugu/tree/main/English` |
| Telugu | `Telugu/checkpoints/` | None: training not completed |

---

## 7. Limitations

- Only about 5 h of training audio per language and 10k steps, so quality is limited.
- English: some clips were lost at alignment and feature extraction (about 3.9 h of the selected 5 h survived). The 8 speakers are demographic groups, so voice identity within a group is not consistent. Heteronyms depend on the g2p front end unless `--fix_heteronyms` is used.
- **Telugu: training was not completed, and checkpoints, loss plots, test audio and variance-adaptor plots are not available.** The Telugu inference and number-handling code paths are untested.
- Telugu design choice: characters are used as phones with no pronunciation rules, which may cause errors with schwa deletion, gemination and loanwords. This is not verified because no Telugu model was evaluated.

## 8. Citation
```bibtex
@inproceedings{ren2021fastspeech2,
  title     = {FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author    = {Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle = {ICLR},
  year      = {2021}
}
```
