# Telugu data preparation files

Source audio and text: IndicTTS Telugu (IIT Madras), via Hugging Face `SPRINGLab/IndicTTS_Telugu`. Please credit IndicTTS / IIT Madras and follow the *License For Use of Indic TTS*.

- `indictts_telugu_split.json`: which recordings are in train and validation
- `telugu_dict.txt`: grapheme pronunciation dictionary (each Telugu character is one sound)
- `TextGrid_Telugu.zip`: forced alignments from Montreal Forced Aligner
- `telugu_acoustic.zip`: MFA acoustic model trained on our data
- `unaligned_utterances.txt`: recordings MFA could not align (skipped)

The audio files are not included. They are kept on Google Drive.
