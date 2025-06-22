# AI CUP DEI - Automatic Speech Recognition (ASR) System
### Fine-tuning OpenAI Whisper and Transformer models for speech recognition

This repository contains comprehensive implementations for the AI CUP DEI Automatic Speech Recognition (ASR) competition. It provides a complete pipeline, including dataset downloading, preprocessing, model training, evaluation, and reproducibility guidelines.

## 📑 Table of Contents
1. [Setup and Installation](#setup-and-installation)
2. [Dataset Download and Preprocessing](#dataset-download-and-preprocessing)
3. [Tasks and Approaches](#tasks-and-approaches)
    - [Task 1: Baseline and Advanced Approaches](#task-1-baseline-and-advanced-approaches)
    - [Task 2: Enhanced ASR Strategies](#task-2-enhanced-asr-strategies)
    - [Differences between Task 1 and Task 2](#differences-between-task-1-and-task-2)
4. [Model and Methods Used](#model-and-methods-used)
5. [Input/Output Formats](#inputoutput-formats)
6. [Training and Reproducibility](#training-and-reproducibility)
7. [Troubleshooting](#troubleshooting)
8. [Detailed Functions and Methods](#detailed-functions-and-methods)
9. [License and Contributors](#license-and-contributors)

## 🚀 Setup and Installation
This project uses Google Colab T4 GPU High-RAM runtime (16 GB RAM).

### Installation
```bash
!pip install transformers==4.50.0 datasets evaluate jiwer librosa accelerate bitsandbytes
!pip install git+https://github.com/openai/whisper.git
```

## 📂 Dataset Download and Preprocessing
Datasets are provided by Codabench, including:

- Sample Dataset
- Training Dataset 1
- Training Dataset 2
- Private Dataset (password-protected)

### Example download and extraction
```python
import requests, zipfile, io

url = "<codabench_url>"
folder = "<your_folder>"
password = b"aicup20660205"

response = requests.get(url)
if response.ok:
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        zf.extractall(folder, pwd=password)
else:
    print("Download failed.")
```

## 📌 Tasks and Approaches

### Task 1: Baseline and Advanced Approaches
- Baseline: Simple Whisper model fine-tuning.
- Approach 1: Vanilla Whisper - direct fine-tuning of Whisper Small.
- Approach 2: Whisper-Medusa - variant with better accuracy and speed.

### Task 2: Enhanced ASR Strategies
- Baseline: Similar to Task 1 baseline.
- Approach 1: Whisper Small fine-tuning.
- Approach 2: Whisper Small + Data Augmentation.
- Approach 3: Whisper Medium fine-tuning.
- Approach 4: Whisper Medium + Multi-stage training.
- Approach 5: Ensemble methods.
- Approach 6: Beam Search with Language Model.

### Differences between Task 1 and Task 2:
- Task 1 focuses on baseline and moderate Whisper enhancements.
- Task 2 explores advanced strategies: augmentation, ensemble, language models.

## 🧩 Model and Methods Used
- Whisper Small / Whisper Medium
- Whisper-Medusa
- DataCollatorSpeechSeq2SeqWithPadding
- Beam Search
- Ensemble techniques
- Prompt template engineering
- Metrics: WER, CER

## 📥 Input/Output Formats

### Inputs
- Audio: `.wav` / `.mp3`, 16kHz
- Transcript: CSV with `file`, `transcript`

### Outputs
- Checkpoints
- Predictions CSV
- WER / CER scores

## ♻️ Training and Reproducibility

### Reproducibility steps
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

Run all notebook cells sequentially: install → download → preprocess → train → evaluate.

## ❗ Troubleshooting

- CUDA OOM: reduce batch size or disable fp16/checkpointing.
- FileNotFound: verify data paths.
- Download fails: check URL and network.

## 📚 Detailed Functions and Methods

| Function/Class | Description |
|----------------|-------------|
| datasets.load_dataset | Load datasets |
| Dataset.cast_column | Cast audio types |
| Dataset.map | Batch preprocessing |
| WhisperProcessor.from_pretrained | Feature extraction & tokenization |
| WhisperForConditionalGeneration | Load Whisper model |
| Seq2SeqTrainer / Seq2SeqTrainingArguments | Training logic |
| evaluate.load("wer") | Word Error Rate |
| evaluate.load("cer") | Character Error Rate |
| DataCollatorSpeechSeq2SeqWithPadding | Batch padding |
| OpenDeidBatchSampler | Custom sampling |
| generate_annotated_audio_transcribe_parallel | Parallel decoding |
| transcribe_with_timestamps | Timestamped decoding |
| collate_batch_with_prompt_template | Prompt formatting |

## 🖋️ License and Contributors

- Contributor: Your Name
- Contact: your.email@example.com
- License: MIT
- Version: AI CUP DEI 2025

## 📮 Contact and Issue Reporting
Contact contributor for questions, suggestions, or bug reports.
