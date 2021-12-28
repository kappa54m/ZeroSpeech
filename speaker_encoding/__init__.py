from .voice_encoder import VoiceEncoder as SpeakerEncoder
from .hparams import *

from pathlib import Path


encoding_size = model_embedding_size


def load_speaker_encoder(device):
    weights_fp = Path(__file__).resolve().parent.joinpath("pretrained.pt")
    speaker_encoder = SpeakerEncoder(device=device, weights_fpath=weights_fp)
    return speaker_encoder


# Original repository: https://github.com/resemble-ai/Resemblyzer
# Resemblyzer is an implementation of GE2E
