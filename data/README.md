# Dataset Documentation

## Overview

This dataset contains multimodal episodic memory data consisting of images and corresponding text captions. The dataset is designed for evaluating Spiking Neural Network (SNN) based memory systems with FAISS indexing for efficient retrieval.

## Dataset Statistics

- **Total Samples**: 1,000+ image-caption pairs
- **Categories**: 9 semantic categories (animals, emotional, human_actions, indoor, objects_tools, outdoor, social_scenes, unusual, others)
- **Image Format**: JPEG, resized to 224×224 pixels
- **Caption Format**: Text descriptions with episodic context

## Directory Structure

```
data/
├── raw/                    # Original, unmodified data
│   ├── images/            # Raw images (various sizes)
│   ├── captions.txt       # Tab-separated: image_name<TAB>caption
│   ├── metadata.json      # Dataset metadata and statistics
│   └── scores.csv         # Quality scores for each sample
├── processed/             # Preprocessed data for model input
│   ├── images_resized/   # Images resized to 224×224
│   ├── spikes/           # Spike-encoded representations
│   │   ├── images/       # Image spike trains (.npy files)
│   │   └── captions/     # Caption spike trains (.npy files)
│   ├── caption_index.json # Mapping: filename → index
│   └── episodes/         # Episodic memory entries
└── README.md             # This file
```

## Data Categories

| Category | Description | Sample Count |
|----------|-------------|--------------|
| animals | Wildlife and pets | ~150 |
| emotional | Emotional expressions | ~12 |
| human_actions | People performing activities | ~100+ |
| indoor | Indoor scenes and objects | ~100+ |
| objects_tools | Tools and manufactured objects | ~100+ |
| outdoor | Outdoor environments | ~100+ |
| social_scenes | People in social contexts | ~100+ |
| unusual | Rare or unusual content | ~100+ |
| others | Miscellaneous | ~100+ |

## Preprocessing Pipeline

### 1. Image Preprocessing (`scripts/preprocess.py`)
- **Input**: Raw images from `data/raw/images/`
- **Operations**:
  - Resize to 224×224 pixels
  - Convert to RGB format
  - Quality validation
- **Output**: `data/processed/images_resized/`

### 2. Spike Encoding (`scripts/encode_spikes.py`)
- **Input**: Resized images + captions
- **Operations**:
  - Rate-based encoding for images
  - Temporal encoding for captions
  - 60 timesteps per sample
- **Output**: Spike trains in `data/processed/spikes/`

### 3. Memory Entry Creation (`scripts/build_memory_entries.py`)
- **Input**: Encoded spikes + metadata
- **Operations**:
  - Combine image and text embeddings
  - Add temporal context
  - Create episodic memory entries
- **Output**: JSONL files in `data/processed/episodes/`

## Data Quality

### Quality Metrics
- **Episodic Relevance**: How well captions describe memorable events
- **Semantic Consistency**: Alignment between image and text
- **Diversity**: Coverage across different categories
- **Quality Scores**: Manual annotation in `scores.csv`

### Quality Distribution
- High quality (>0.8): 70%
- Medium quality (0.5-0.8): 25%
- Low quality (<0.5): 5%

## Usage in Experiments

### Training Data
```python
# Load spike-encoded pairs
dataset = SpikeEpisodeDataset(
    img_dir="data/processed/spikes/images/",
    cap_dir="data/processed/spikes/captions/",
    index_json="data/processed/caption_index.json"
)
```

### Memory Entries
```python
# Load episodic memory entries
with open("data/processed/episodes/episodes.jsonl", "r") as f:
    for line in f:
        episode = json.loads(line)
        # Process episode with temporal context
```

## Ethical Considerations

- **Content Filtering**: Dataset contains only appropriate content for research
- **Privacy**: No personal identifiable information
- **Bias**: Balanced representation across categories
- **Usage**: Intended for academic research only

## Citation

When using this dataset in research, please cite:

```
@dataset{snn_faiss_dataset,
  title={SNN-FAISS Episodic Memory Dataset},
  author={Research Team},
  year={2024},
  description={Multimodal dataset for evaluating spiking neural network memory systems}
}
```

## Maintenance

- **Version**: 1.0
- **Last Updated**: December 2024
- **Contact**: For dataset updates or issues
