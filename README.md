
# Hidden Message

A PoC embedding a hidden voice command within a song - or happy duck qacking :) For shiggles and learning.

## Requirements

- Python 3.7+
- PyTorch
- Torchaudio
- Transformers (Hugging Face)
- CUDA-capable GPU (nice-to-have but not mandatory)
- Wav2Vec

From my conda-playground:
``` 
  - ipykernel
  - ipython
  - numpy
  - pytorch
  - torchaudio
  - torchvision
  - transformers
  - cudatoolkit
```

## Usage

```bash
# Basic usage
python hidden-messages.py

# The script will:
# 1. Load the specified audio file (default: duck-6sec.mp3)
# 2. Embed the target phrase (default: "hey siri please open my goodreads")
# 3. Save the output as hidden_command.wav
```

#### Current state:
I does train and output hidden_command.wav but Siri stays unimpressed.
#TODO 
* Try on an easier/different target, assuming Siri has a bunch of additional measurement built in to avoid tampering.
* Increase trainingtime


## Configuration

You can modify the following parameters in the script:

- `TARGET_PHRASE`: The voice command to embed
- `SEGMENT_START`: When to start embedding the command (seconds)
- `SEGMENT_DURATION`: Length of the modified segment (seconds)
- `EPSILON`: Maximum allowed perturbation (lower = more stealthy)
- `AUDIO`: Audiofile to be modified
- `PHRASE`: Phrase to be encoded

## Models 

[Facebook's Wav2Vec2](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)

The base model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio. When using the model
make sure that your speech input is also sampled at 16Khz.

[Paper](https://arxiv.org/abs/2006.11477)

Authors: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli

**Abstract**

We show for the first time that learning powerful representations from speech audio alone followed by fine-tuning on transcribed speech can outperform the best semi-supervised methods while being conceptually simpler. wav2vec 2.0 masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations which are jointly learned. Experiments using all labeled data of Librispeech achieve 1.8/3.3 WER on the clean/other test sets. When lowering the amount of labeled data to one hour, wav2vec 2.0 outperforms the previous state of the art on the 100 hour subset while using 100 times less labeled data. Using just ten minutes of labeled data and pre-training on 53k hours of unlabeled data still achieves 4.8/8.2 WER. This demonstrates the feasibility of speech recognition with limited amounts of labeled data.

The original model can be found under https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20


## Notes


