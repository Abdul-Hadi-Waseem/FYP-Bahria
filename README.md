# Deep Learning for Lung Sound Analysis

This repository contains the implementation of deep learning methods for lung sound analysis using wireless stethoscopes, as described in our paper "Deep learning in wireless stethoscope-based lung sound analysis".

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Dataset](#dataset)

## Overview

A comprehensive toolkit for analyzing lung sounds using deep learning, built with:

- PyTorch
- Librosa
- pyAudioAnalysis

## Requirements

- Python ≥ 3.10
- Additional dependencies listed in `pyproject.toml`

## Installation

1. Install uv:

#### Linux/MacOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Clone and setup the project:

```bash
git clone git@github.com:Abdul-Hadi-Waseem/FYP-Bahria.git
cd FYP-Bahria

# Create venv and install dependencies
uv venv
uv pip install .
```

### Quick Start

Run the complete pipeline:

```bash
uv run run_pipeline.py
```

## Pipeline Components

### 1. Preprocessing (`preprocessing.py`)

- Noise reduction
- Data segmentation
- Supported denoising methods:
  - Bandpass filtering (recommended for respiratory sounds)
  - Wavelet denoising
  - EMD-based denoising

### 2. Feature Extraction (`feature_extraction.py`)

- Statistical features (mean, variance, skewness, etc.)
- Mel-spectrogram generation (default: 128 mel bands)
- Standard spectrogram creation (STFT-based)

### 3. Data Splitting (`data_splitting.py`)

- Subject-wise cross-validation
- Stratified k-fold splitting

### 4. Model Training (`Classifier.py`)

- Multiple model architectures
- Training and evaluation pipeline
- Performance metrics calculation

## Configuration

All parameters can be customized in `Config.py`, including:

- Audio processing parameters
- Model architecture settings
- Training hyperparameters
- Cross-validation settings

## Dataset

The implementation is demonstrated using the ICBHI 2017 dataset. The pipeline can be adapted for other lung sound datasets.
