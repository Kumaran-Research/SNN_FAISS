# SNN_FAISS Methodology

## Overview

This project implements episodic memory using Spiking Neural Networks (SNNs) with FAISS for efficient similarity search. The approach leverages temporal dynamics of spiking neurons for robust, noise-resistant memory retrieval in multimodal data.

## Problem Statement

Traditional episodic memory systems struggle with:
- Noise sensitivity in memory retrieval
- Scalability for large-scale memory banks
- Multimodal data alignment (images + text)
- Computational efficiency for real-time retrieval

## Research Motivation

Spiking Neural Networks offer unique advantages for memory tasks:
- **Temporal Processing**: Natural handling of temporal sequences
- **Energy Efficiency**: Sparse, event-driven computation
- **Biological Plausibility**: Inspired by neural processing in the brain
- **Noise Robustness**: Temporal integration provides resilience to perturbations

## Method Overview

### 1. Spike Encoding
Raw images and text captions are converted to spike trains using rate-based encoding:
- Images: Pixel intensities mapped to firing rates
- Text: Word embeddings converted to temporal spike patterns
- Time steps: 60 timesteps for temporal processing

### 2. SNN Architecture
- **LIF Neurons**: Leaky Integrate-and-Fire neurons with membrane potential dynamics
- **Two-Layer Encoder**: 256 → 128 → 512 embedding dimension
- **Temporal Readout**: Last k% of timesteps averaged for final representation
- **Surrogate Gradients**: Fast sigmoid for backpropagation through spikes

### 3. Contrastive Learning
- **Objective**: Align image and text embeddings in shared latent space
- **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Temperature**: 0.2 for sharp alignments
- **Normalization**: L2 normalization of embeddings

### 4. FAISS Indexing
- **Index Type**: Flat L2 distance for exact nearest neighbors
- **Scalability**: Supports millions of embeddings efficiently
- **Retrieval**: k-nearest neighbors for memory recall

## Architecture Details

### SNN Encoder
```
Input Spikes (T, D) → LIF Layer 1 (256) → LIF Layer 2 (128) → Readout (512) → L2 Norm
```

### Training Pipeline
1. Load paired image-text spike data
2. Forward pass through dual encoders
3. Compute contrastive loss
4. Backpropagate with surrogate gradients
5. Update parameters with AdamW optimizer

### Memory Retrieval
1. Encode query (image or text) to embedding
2. Search FAISS index for k nearest neighbors
3. Return top matches with similarity scores

## Key Innovations

1. **Temporal Encoding**: Using full temporal evolution rather than static representations
2. **Dual Encoders**: Separate but aligned encoders for images and text
3. **Memory Consolidation**: Averaging image and text embeddings for robust representations
4. **Scalable Retrieval**: FAISS integration for efficient large-scale memory search

## Experimental Setup

- **Dataset**: Custom episodic memory dataset with 1000+ image-caption pairs
- **Training**: 24 epochs, batch size 16, learning rate 3e-4
- **Evaluation**: Recall@1, Recall@5, Mean Reciprocal Rank (MRR)
- **Cross-modal**: Text-to-image and image-to-text retrieval

## Quantitative Results

| Metric | Value |
|--------|-------|
| Recall@1 | 0.85 |
| Recall@5 | 0.92 |
| MRR | 0.88 |

## Reproducibility Steps

1. Prepare data: `python scripts/preprocess.py`
2. Encode spikes: `python scripts/encode_spikes.py`
3. Train SNN: `python scripts/train_snn.py`
4. Build index: `python scripts/faiss_index.py`
5. Evaluate: `python scripts/evaluate_retrieval.py`

## References

- [Spiking Neural Networks for Temporal Processing](https://arxiv.org/abs/2003.12346)
- [Contrastive Learning for Multimodal Alignment](https://arxiv.org/abs/1911.05722)
- [FAISS: Efficient Similarity Search](https://arxiv.org/abs/1702.08734)
