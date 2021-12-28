import hydra
from hydra import utils
from pathlib import Path
import librosa
import scipy
import json
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

from util import fix_config


def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)


def mulaw_encode(x, mu):
    mu = mu - 1
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.floor((fx + 1) / 2 * mu + 0.5)


def mulaw_decode(y, mu):
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


def load_wav(wav_path, sr=160000, offset=0.0, duration=None):
    wav, _ = librosa.load(str(wav_path.with_suffix(".wav")), sr=sr,
                          offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999
    return wav


def process_wav(wav_path, out_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav = load_wav(wav_path.with_suffix(".wav"), sr=sr, offset=offset, duration=duration)

    mel = librosa.feature.melspectrogram(preemphasis(wav, preemph),
                                         sr=sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         hop_length=hop_length,
                                         win_length=win_length,
                                         fmin=fmin,
                                         power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    wav = mulaw_encode(wav, mu=2**bits)

    np.save(out_path.with_suffix(".wav.npy"), wav)
    np.save(out_path.with_suffix(".mel.npy"), logmel)
    return out_path, logmel.shape[-1]


def process_for_speaker_encoding(wav_path_or_paths, out_path, embed_utterance=True, sr=160000, offset=0.0, duration=None):
    if embed_utterance:
        wav = load_wav(wav_path_or_paths.with_suffix(".wav"), sr=sr, offset=offset, duration=duration)
        enc = speaker_encoder.embed_utterance(wav)
    else:
        wavs = []
        for i, wav_path in enumerate(wav_path_or_paths):
            wav = load_wav(wav_path.with_suffix(".wav"), sr=sr, offset=offset[i], duration=duration[i])
            wavs.append(wav)
        enc = speaker_encoder.embed_speaker(wavs)

    np.save(out_path.with_suffix(".enc.npy"), enc)
    return out_path


@hydra.main(config_path="config", config_name="preprocessing.yaml")
def preprocess_dataset(cfg):
    cfg = fix_config(cfg)

    in_dir = Path(utils.to_absolute_path(cfg.in_dir))
    out_dir = Path(utils.to_absolute_path("datasets")) / str(cfg.dataset.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.preprocessing.encode_speakers:
        from speaker_encoding import load_speaker_encoder
        global speaker_encoder
        speaker_encoder = load_speaker_encoder(None)

    executor = ProcessPoolExecutor(max_workers=cpu_count())
    for split in ["train", "test"]:
        print("Extracting features for {} set".format(split))
        futures = []
        speaker_encoding_tasks = []
        split_path = out_dir / cfg.dataset.language / split
        with open(split_path.with_suffix(".json")) as file:
            metadata = json.load(file)
            for in_path, start, duration, out_path in metadata:
                wav_path = in_dir / in_path
                out_path = out_dir / out_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                process_wav_args = {k: v for k, v in cfg.preprocessing.items() if k != 'encode_speakers'}
                futures.append(executor.submit(
                    partial(process_wav, wav_path, out_path, **process_wav_args,
                            offset=start, duration=duration)))
                if cfg.preprocessing.encode_speakers:
                    speaker_encoding_tasks.append({
                        'wav_path': wav_path,
                        'out_path': out_path,
                        'offset': start,
                        'duration': duration,
                        'speaker': out_path.parts[-2],
                    })

        results = [future.result() for future in tqdm(futures)]

        # Preprocess for speaker encoding
        if cfg.preprocessing.encode_speakers:
            print("Computing speaker embedding per sample...")
            for d in tqdm(speaker_encoding_tasks):
                process_for_speaker_encoding(
                    d['wav_path'], d['out_path'], embed_utterance=True,
                    offset=d['offset'], duration=d['duration'], sr=cfg.preprocessing.sr)

            print("Computing speaker embedding per speaker...")
            speakers = list(set([v['speaker'] for v in speaker_encoding_tasks]))
            s_out_dir = split_path / "speakers"
            s_out_dir.mkdir(parents=True, exist_ok=True)
            for speaker in tqdm(speakers):
                p = {k: [] for k in ['wav_path_or_paths', 'offset', 'duration']}
                for d in speaker_encoding_tasks:
                    if d['speaker'] == speaker:
                        p['wav_path_or_paths'].append(d['wav_path'])
                        p['offset'].append(d['offset'])
                        p['duration'].append(d['duration'])
                out_path = s_out_dir / speaker
                process_for_speaker_encoding(**p, out_path=out_path,
                                             embed_utterance=False, sr=cfg.preprocessing.sr)

        lengths = [x[-1] for x in results]
        frames = sum(lengths)
        frame_shift_ms = cfg.preprocessing.hop_length / cfg.preprocessing.sr
        hours = frames * frame_shift_ms / 3600
        print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


if __name__ == "__main__":
    preprocess_dataset()
