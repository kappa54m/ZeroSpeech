import numpy as np
import torch
from torch.utils.data import Dataset
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sr, sample_frames, encoded_speakers, speaker_encoding_opts=None):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames
        self.encoded_speakers = encoded_speakers
        if encoded_speakers:
            opts = speaker_encoding_opts or {}
            self.speaker_encoding_per_speaker = opts.get('per_speaker', False)
            if self.speaker_encoding_per_speaker:
                self.speaker_encoding_speakers_dir_rel = opts['speakers_dir']
        else:
            self.speaker_encoding_per_speaker = False
            self.speaker_encoding_speakers_dir_rel = None
        self.speaker_encoding_speakers_dir_base = None

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sr
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
            self.metadata = [
                Path(out_path) for _, _, duration, out_path in metadata
                if duration > min_duration
            ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root.parent / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker_id = path.parts[-2]
        if self.encoded_speakers:
            if self.speaker_encoding_per_speaker:
                speakers_dir = self.speaker_encoding_speakers_dir_base / self.speaker_encoding_speakers_dir_rel
                enc_path = (self.speakers_dir / speaker_id).with_suffix(".enc.npy")
            else:
                enc_path = path.with_suffix(".enc.npy")
            speaker_enc = np.load(enc_path)
            speaker = torch.FloatTensor(speaker_enc)
        else:
            speaker = self.speakers.index(speaker_id)

        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker
