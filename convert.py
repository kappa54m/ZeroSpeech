import hydra
import hydra.utils as utils

import json
from pathlib import Path
import torch
import numpy as np
import librosa
import soundfile
from tqdm import tqdm
import pyloudnorm

from preprocess import preemphasis
from model import Encoder, Decoder
from util import fix_config


@hydra.main(config_path="config", config_name="convert.yaml")
def convert(cfg):
    cfg = fix_config(cfg)
    dataset_path = Path(utils.to_absolute_path("datasets")) / cfg.dataset.path
    with open(dataset_path / "speakers.json") as file:
        speakers = sorted(json.load(file))

    synthesis_list_path = Path(utils.to_absolute_path(cfg.synthesis_list))
    with open(synthesis_list_path) as file:
        synthesis_list = json.load(file)

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path(cfg.out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_encoder = None
    if not cfg.model.speaker_embedding.use_basic_speaker_embedding:
        from speaker_encoding import load_speaker_encoder
        speaker_encoder = load_speaker_encoder(device)

    encoder = Encoder(**cfg.model.encoder)
    decoder = Decoder(**cfg.model.decoder)
    encoder.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    meter = pyloudnorm.Meter(cfg.preprocessing.sr)

    def load_wav(_path):
        _wav_path = in_dir / _path
        _wav, _ = librosa.load(
            str(_wav_path.with_suffix(".wav")),
            sr=cfg.preprocessing.sr)
        _ref_loudness = meter.integrated_loudness(_wav)
        _wav = _wav / np.abs(_wav).max() * 0.999
        return _wav, _ref_loudness

    for wav_path, speaker_id, out_filename in tqdm(synthesis_list):
        wav, ref_loudness = load_wav(wav_path)

        mel = librosa.feature.melspectrogram(
            preemphasis(wav, cfg.preprocessing.preemph),
            sr=cfg.preprocessing.sr,
            n_fft=cfg.preprocessing.n_fft,
            n_mels=cfg.preprocessing.n_mels,
            hop_length=cfg.preprocessing.hop_length,
            win_length=cfg.preprocessing.win_length,
            fmin=cfg.preprocessing.fmin,
            power=1)
        logmel = librosa.amplitude_to_db(mel, top_db=cfg.preprocessing.top_db)
        logmel = logmel / cfg.preprocessing.top_db + 1

        mel = torch.FloatTensor(logmel).unsqueeze(0).to(device)
        if speaker_encoder is not None:
            if speaker_id.endswith(".wav"):
                speaker_wav_path = in_dir / speaker_id
                print("Computing speaker encoding from: {}".format(speaker_wav_path))
                enc = speaker_encoder.embed_utterance(load_wav(speaker_wav_path)[0])
                speaker = torch.FloatTensor(enc)
            elif speaker_id.endswith(".enc.npy"):
                enc_path = in_dir / speaker_id
                print("Loading speaker encoding from: {}".format(enc_path))
                speaker = torch.FloatTensor(np.load(enc_path))
            else:
                encs_dir = Path(cfg.speaker_encodings_dir)
                if not encs_dir.is_absolute():
                    encs_dir = utils.to_absolute_path(str(encs_dir))
                enc_path = next(filter(lambda s: s.name == "{}.enc.npy".format(speaker_id), Path(encs_dir).iterdir()), None)
                if enc_path:
                    print("Found encoding for speaker '{}' at: {}. Loading...".format(speaker_id, enc_path))
                    speaker = torch.FloatTensor(np.load(enc_path))
                else:
                    raise ValueError(
                        "Could not find encoding for speaker '{}' in: {}. "
                        "Make sure speaker_encodings_dir argument (current value: \"{}\") is set correctly.".format(
                            speaker_id, encs_dir, cfg.speaker_encodings_dir))
        else:
            speaker = torch.LongTensor([speakers.index(speaker_id)])
        speaker = speaker.to(device)

        with torch.no_grad():
            z, _ = encoder.encode(mel)
            output = decoder.generate(z, speaker)

        output_loudness = meter.integrated_loudness(output)
        output = pyloudnorm.normalize.loudness(output, output_loudness, ref_loudness)
        path = out_dir / out_filename
        soundfile.write(str(path.with_suffix(".wav")), output.astype(np.float32), samplerate=cfg.preprocessing.sr)


if __name__ == "__main__":
    convert()
