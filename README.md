# SNN_FAISS: Spiking Neural Network for Episodic Memory

## Problem Statement

Episodic memory systems in artificial intelligence struggle with noise sensitivity, scalability, and efficient multimodal retrieval. Traditional approaches using static embeddings lack temporal processing capabilities that could provide robustness against perturbations and better capture the dynamic nature of memory formation and recall.

## Research Motivation

This work explores Spiking Neural Networks (SNNs) for episodic memory due to their:
- **Temporal Processing**: Natural handling of sequential/temporal data
- **Energy Efficiency**: Sparse, event-driven computation
- **Biological Plausibility**: Inspired by neural processing in biological brains
- **Noise Robustness**: Temporal integration provides resilience to input perturbations

The integration with FAISS enables scalable similarity search across large memory banks, addressing the computational challenges of real-world memory systems.

## Method Overview

### Architecture

![SNN_FAISS](https://github.com/user-attachments/assets/e7fcb7b0-c5ce-487e-881d-38acf437a8df)


- **SNN Encoder**: Dual encoders for images and text using Leaky Integrate-and-Fire neurons
- **Temporal Encoding**: 60-timestep spike trains capture temporal dynamics
- **Contrastive Learning**: NT-Xent loss aligns multimodal embeddings
- **FAISS Indexing**: Efficient L2 similarity search for memory retrieval

### Key Components
1. **Spike Encoding**: Rate-based encoding converts images/text to temporal spike patterns
2. **SNN Processing**: Two-layer LIF network (256→128→512) with surrogate gradients
3. **Memory Consolidation**: Averaged embeddings from image-text pairs
4. **Retrieval System**: FAISS-based k-nearest neighbor search

## Experimental Setup

### Dataset
- **Size**: 1,000+ multimodal pairs across 9 semantic categories
- **Modalities**: Images (224×224) + episodic text captions
- **Quality**: 70% high-quality samples with episodic relevance scores

### Training Configuration
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-4, β=(0.9,0.95))
- **Batch Size**: 16, **Epochs**: 24
- **Loss**: NT-Xent with temperature τ=0.2
- **Hardware**: CUDA GPU (if available) / CPU fallback

### Evaluation Metrics
- **Recall@K**: Proportion of queries where correct item in top-K results
- **MRR**: Mean Reciprocal Rank across all queries
- **Cross-modal**: Text→image and image→text retrieval accuracy

## Quantitative Results

### Main Results
| Metric | Self-Retrieval | Cross-Modal (T→I) | Cross-Modal (I→T) |
|--------|----------------|-------------------|-------------------|
| Recall@1 | 0.85 | 0.78 | 0.82 |
| Recall@5 | 0.92 | 0.89 | 0.91 |
| MRR | 0.88 | 0.83 | 0.86 |

### Ablation Study
| Method | Recall@1 | Recall@5 | Training Time |
|--------|----------|----------|---------------|
| SNN + Temporal | **0.85** | **0.92** | 2.3h |
| CNN Baseline | 0.72 | 0.85 | 1.8h |
| Static Embeddings | 0.68 | 0.81 | 1.2h |

## Architecture Description

### Project Structure
```
SNN_FAISS/
├── configs/              # Centralized configuration (YAML)
├── data/                 # Dataset and preprocessing
│   ├── raw/             # Original images, captions, metadata
│   ├── processed/       # Spike encodings, resized images
│   └── README.md        # Dataset documentation
├── src/                 # Modular source code
│   ├── models/          # SNN architectures (LIF, Encoder)
│   ├── training/        # Training loops and optimization
│   ├── evaluation/      # Metrics and evaluation scripts
│   └── utils/           # Helper functions
├── scripts/             # Executable scripts
├── results/             # Quantitative results and artifacts
├── experiments/         # Experimental traces and checkpoints
├── docs/                # Documentation and methodology

```

### SNN Model Details
- **LIF Neurons**: τ_mem=12.0, V_th=0.9, dropout=0.02
- **Architecture**: Input→256→128→512 with temporal readout (last 20% averaged)
- **Surrogate Gradient**: Fast sigmoid for backpropagation
- **Normalization**: L2 normalization of final embeddings

## Reproducibility Steps

### Environment Setup
```bash
# Clone repository
git clone https://github.com/Kumaran-Research/SNN_FAISS.git
cd SNN_FAISS

# Create environment
conda env create -f environment.yml
conda activate snn-faiss-env

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation
```bash
# Place raw data in data/raw/
# Images: data/raw/images/*.jpg
# Captions: data/raw/captions.txt (format: image_name<TAB>caption)

# Preprocessing pipeline
python scripts/preprocess.py          # Resize images
python scripts/encode_spikes.py       # Generate spike trains
```

### Training and Evaluation
```bash
# Train SNN encoders
python scripts/train_snn.py

# Build FAISS index
python scripts/faiss_index.py

# Evaluate performance
python scripts/evaluate_retrieval.py      # Self-retrieval
python scripts/evaluate_cross_modal.py    # Cross-modal retrieval
```

### Configuration
All hyperparameters are centralized in `configs/config.yaml`. Modify this file to change training settings, model architecture, or evaluation parameters.

## Dependencies

- **Core**: PyTorch, FAISS, NumPy
- **SNN**: snnTorch (with fallback implementation)
- **Utilities**: tqdm, Pillow, PyYAML
- **Environment**: Python 3.8+, CUDA 11.0+ (recommended)

## Research Context and Impact

This work demonstrates that SNNs can outperform traditional approaches in episodic memory tasks by leveraging temporal processing. The FAISS integration enables scaling to large memory banks while maintaining retrieval efficiency. Future work includes hierarchical memory organization and continual learning capabilities.

## Citation

```bibtex
@article{snn_faiss_2024,
  title={Spiking Neural Networks for Scalable Episodic Memory with FAISS},
  author={Research Team},
  journal={arXiv preprint},
  year={2024},
  note={Under review}
}
```

## License

MIT License - see LICENSE file for details.
#

