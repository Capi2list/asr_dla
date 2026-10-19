# Automatic Speech Recognition (ASR) with Deep Speech

<p align="center">
  <a href="#about">About</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository based on [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template). It contains the implementation of model Deep Speech 2 for ASR Task, based on the article [Deep Speech 2: End-to-End Speech Recognition in
English and Mandarin](https://arxiv.org/pdf/1512.02595). Repo contains the demo notebook, so you can easily deploy and test model perfomance on your own dataset or on LibiriSpeech.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).
## Metrics
Repo contains simple hand-crafted beam search for the evaluation. Also for evaluation was added LM + beam search from [pyctcdecode](https://github.com/kensho-technologies/pyctcdecode) library. Note that it's recommended to use this option, instead of hand-made beam_search because of significant quality and speed increase.

The model was trained on LibiriSpeech dataset (train-clean-100 + train-clean-360 + train-other-500) using `train.yaml` config. The decoding method is indicated in parentheses, e.g. argmax means we took the character with highest probability.

| Dataset                            | CER     | WER     |
|------------------------------------|---------|---------|
| test_clean (argmax)                | 0.04589 | 0.15494 |
| test_other (argmax)                | 0.13887 | 0.36502 |
| test_other (hand-made beam search) | 0.04487 | 0.15095 |
| test_other (hand-made beam search) | 0.13700 | 0.36011 |
| test_clean (LM beam search)        | 0.03120 | 0.08560 |
| test_other (LM beam search)        | 0.11035 | 0.23765 |


## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install pre-commit:

   ```bash
   pre-commit install
   ```


## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To run inference (evaluate the model or save predictions):

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
You can download model checkpoints and LM model using `download_data.py`:

```bash
python3 src/scripts/download_data.py --output_dir OUTPUT_DIR
```
To calculate CER/WER metrics on inference.py outputs use `calc_metrics.py`:

```bash
python3 calc_metrics.py --input_dir INPUT_DIR
```

## Demo

Repo contains `demo.ipynb`. It shows demo of a project, i.e., showcases how to use it.

## Examples

If you want to replicate the best model checkpoint use `train.yaml`:

```bash
python3 train.py -cn=train
```

If you want to run inference with different encoders (BeamSearchEncoder, CTCTextEncoder (argmax one), LMTextEncoder) run:

```bash
python3 inference.py -cn=inference text_encoder._target_=src.text_encoder.CTCTextEncoder
```

If you want to change model params, you can do it this way, however it's better to change configs.

```bash
python3 train.py -cn=train model.hidden_size=768 model.num_layers=5
```

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
