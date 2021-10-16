# VQ-VAE for Acoustic Unit Discovery and Voice Conversion

Train and evaluate the VQ-VAE model for our submission to the [ZeroSpeech 2020 challenge](https://zerospeech.com/).
Voice conversion samples can be found [here](https://bshall.github.io/ZeroSpeech/).
Pretrained weights for the 2019 English and Indonesian datasets can be found [here](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1).

<div align="center">
    <img width="495" height="639" alt="VQ-VAE for Acoustic Unit Discovery" 
      src="https://raw.githubusercontent.com/bshall/ZeroSpeech/master/model.png"><br>
    <sup><strong>Fig 1:</strong> VQ-VAE model architecture.</sup>
</div>

# Quick Start

## Requirements

1.  Ensure you have Python 3 and PyTorch 1.4 or greater.

2.  Install [NVIDIA/apex](https://github.com/NVIDIA/apex) for mixed precision training.

3.  Install pip dependencies:
    ```
    pip install -r requirements.txt
    ```

4.  For evaluation install [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020).

## Data and Preprocessing

1.  Download and extract the [ZeroSpeech2020 datasets](https://download.zerospeech.com/).

2.  Download the train/test splits [here](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1) 
    and extract in the root directory of the repo.
    
3.  Preprocess audio and extract train/test log-Mel spectrograms:
    ```
    python preprocess.py in_dir=/path/to/dataset dataset=[2019/english or 2019/surprise]
    ```
    Note: `in_dir` must be the path to the `2019` folder. 
    For `dataset` choose between `2019/english` or `2019/surprise`.
    Other datasets will be added in the future.
    ```
    e.g. python preprocess.py in_dir=../datasets/2020/2019 dataset=2019/english
    ```
    
## Training
   
Train the models or download pretrained weights [here](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1):
```
python train.py checkpoint_dir=path/to/checkpoint_dir dataset=[2019/english or 2019/surprise]
```
```
e.g. python train.py checkpoint_dir=checkpoints/2019english dataset=2019/english
```
   
## Evaluation
    
### Voice conversion

```
python convert.py checkpoint=path/to/checkpoint in_dir=path/to/wavs out_dir=path/to/out_dir synthesis_list=path/to/synthesis_list dataset=[2019/english or 2019/surprise]
```
Note: the `synthesis list` is a `json` file:
```
[
    [
        "english/test/S002_0379088085",
        "V002",
        "V002_0379088085"
    ]
]
```
containing a list of items with a) the path (relative to `in_dir`) of the source `wav` files;
b) the target speaker (see `datasets/2019/english/speakers.json` for a list of options);
and c) the target file name.
```
e.g. python convert.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt in_dir=../datasets/2020/2019 out_dir=submission/2019/english/test synthesis_list=datasets/2019/english/synthesis.json dataset=2019/english
```
Voice conversion samples can be found [here](https://bshall.github.io/ZeroSpeech/).

### ABX Score
    
1.  Encode test data for evaluation:
    ```
    python encode.py checkpoint=path/to/checkpoint out_dir=path/to/out_dir dataset=[2019/english or 2019/surprise]
    ```
    ```
    e.g. python encode.py checkpoint=checkpoints/2019english/model.ckpt-500000.pt out_dir=submission/2019/english/test dataset=2019/english
    ```
    
2. Run ABX evaluation script (see [bootphon/zerospeech2020](https://github.com/bootphon/zerospeech2020)).

The ABX score for the pretrained english model (available [here](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1)) is:
```
{
    "2019": {
        "english": {
            "scores": {
                "abx": 14.043611615570672,
                "bitrate": 412.2387509949519
            },
            "details_bitrate": {
                "test": 412.2387509949519
            },
            "details_abx": {
                "test": {
                    "cosine": 14.043611615570672,
                    "KL": 50.0,
                    "levenshtein": 35.927825062038984
                }
            }
        }
    }
}
```

# Speaker Encoding
## Quick Start
1. [bshall/ZeroSpeech Releases](https://github.com/bshall/ZeroSpeech/releases/tag/v0.1)에서 `datasets.zip`을 다운받아서 프로젝트 폴더 안에 압축 해제합니다($PROJECTROOT/datasets/2019/...).
2. 기존 전처리에 추가로 Speaker Encoder의 벡터를 전처리합니다.

- Speaker Encoding 활성화
```
python preprocess.py in_dir=path/to/dataset dataset=2019/english preprocessing.preprocessing.encode_speakers=true
```

- 기존 모델
```
python preprocess.py in_dir=path/to/dataset dataset=2019/english preprocessing.preprocessing.encode_speakers=false
```

3. 트레이닝

- Speaker Encoding 활성화
```
python train.py checkpoint_dir=path/to/checkpoint_dir dataset=2019/english model.model.speaker_embedding.use_basic_speaker_embedding=false model.model.speaker_embedding.options.per_speaker=false
```

- 기존 모델
```
python train.py checkpoint_dir=path/to/checkpoint_dir dataset=2019/english model.model.speaker_embedding.use_basic_speaker_embedding=true
```

4. 테스트

- Speaker Encoding 활성화
```
python convert.py checkpoint=path/to/checkpoint in_dir=path/to/wavs out_dir=path/to/out_dir synthesis_list=path/to/synthesis_list dataset=2019/english model.model.speaker_embedding.use_basic_speaker_embedding=false model.model.speaker_embedding.options.per_speaker=false
```

`synthesis_list`의 두 번째 값으로 기존에는 화자 스트링(예: V001)을 주었는데, 이 모델의 경우 화자의 인코딩이 필요하기 때문에 이 경우에는 추가로 `speaker_encodings_dir` 인자를 추가해야 하는데, 값은 프로젝트 폴더에서의 상대적 경로 또는 절대적 경로로 설정하면 됩니다.
화자 스트링 대신 `wav` 파일의 경로나 전처리에서 생성되는 `.enc.npy` 파일의 경로를 입력할 수 있습니다. 상대적 경로일 경우 `in_dir` 인자를 부모 폴더로 합니다.

- 기존 모델

```
python convert.py checkpoint=path/to/checkpoint in_dir=path/to/wavs out_dir=path/to/out_dir synthesis_list=path/to/synthesis_list dataset=2019/english model.model.speaker_embedding.use_basic_speaker_embedding=true
```

# References

This work is based on:

1.  Chorowski, Jan, et al. ["Unsupervised speech representation learning using wavenet autoencoders."](https://arxiv.org/abs/1901.08810)
    IEEE/ACM transactions on audio, speech, and language processing 27.12 (2019): 2041-2053.

2.  Lorenzo-Trueba, Jaime, et al. ["Towards achieving robust universal neural vocoding."](https://arxiv.org/abs/1811.06292)
    INTERSPEECH. 2019.
    
3.  van den Oord, Aaron, and Oriol Vinyals. ["Neural discrete representation learning."](https://arxiv.org/abs/1711.00937)
    Advances in Neural Information Processing Systems. 2017.

## 연구 논문들
1. [bshall/ZeroSpeech](https://github.com/bshall/ZeroSpeech) 모델 전체: Chorowski, Jan, et al. ["Unsupervised speech representation learning using wavenet autoencoders."](https://arxiv.org/abs/1901.08810)
2. bshall/ZeroSpeech 모델에 사용된 디코더 WaveNet 모델: Lorenzo-Trueba, Jaime, et al. ["Towards achieving robust universal neural vocoding."](https://arxiv.org/abs/1811.06292)
3. bshall/ZeroSpeech 모델의 디코더와 최첨단 오디오 합성의 기본이 되는 [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio): ["WaveNet: A Generative Model for Raw Audio"](https://arxiv.org/pdf/1609.03499.pdf)
4. WaveNet의 기본이 되는 [Autoregressive 신경망](https://ml.berkeley.edu/blog/posts/AR_intro/) PixelRNN/PixelCNN:
- ["Pixel Recurrent Neural Networks"](https://arxiv.org/abs/1601.06759)
- ["Conditional Image Generation with PixelCNN Decoders"](https://arxiv.org/abs/1606.05328)
5. bshall/ZeroSpeech 모델의 인코더에 사용되는 오토인코더 [VQ-VAE](https://ml.berkeley.edu/blog/posts/vq-vae/): van den Oord, Aaron, and Oriol Vinyals. ["Neural discrete representation learning."](https://arxiv.org/abs/1711.00937)
