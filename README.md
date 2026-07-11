# Wheat Disease Classification

CNN-based deep learning model for automated detection and classification of wheat diseases from leaf images. The model identifies **15 different classes** including various diseases and healthy plant samples.

## Features

- **15-class classification**: Aphid, Black Rust, Blast, Brown Rust, Common Root Rot, Fusarium Head Blight, Healthy, Leaf Blight, Mildew, Mite, Septoria, Smut, Stem fly, Tan spot, Yellow Rust
- **Target-driven training**: Automatically stops when all classes reach 95% recall
- **GPU memory-safe**: Includes memory management to prevent OOM errors on long training runs
- **Comprehensive reporting**: Generates 12+ visualizations including confusion matrix, ROC curves, PR curves, training history, and class performance metrics
- **Checkpoint management**: Automatic saving and cleanup of model checkpoints

## Model Architecture

The model uses a custom 5-block CNN with the following structure:

| Block | Filters | Dropout |
|-------|---------|---------|
| Block 1 | 32 | 0.25 |
| Block 2 | 64 | 0.25 |
| Block 3 | 128 | 0.30 |
| Block 4 | 256 | 0.30 |
| Block 5 | 512 | 0.40 |
| Classifier | 512 + 256 | 0.50 |

- **Total parameters**: ~5.1M
- **Input size**: 224 x 224 x 3
- **Regularization**: L2 (1e-4) on all conv/dense layers + BatchNorm

## Dataset Structure

```
D:\dataset\wheat_desease\
├── train\
│   ├── Aphid\
│   ├── Black Rust\
│   ├── Blast\
│   ├── ...
│   └── Yellow Rust\
└── valid\
    ├── Aphid\
    ├── Black Rust\
    ├── Blast\
    ├── ...
    └── Yellow Rust\
```

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- See `requirements.txt` for full dependencies

## Usage

1. **Prepare your dataset** in the structure shown above
2. **Update the path** in `cli.py`:
   ```python
   DATASET_ROOT = r'D:\dataset\wheat_desease'
   ```
3. **Run training**:
   ```bash
   python cli.py
   ```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Initial LR | 0.001 |
| Min LR | 1e-7 |
| Target Recall | 95% per class |
| Max Epochs | 200 |
| Data Augmentation | Rotation, shift, shear, zoom, flip, brightness |

## Output

All outputs are saved to the `outputs/` directory:

```
outputs/
├── checkpoints/       # Model checkpoints (epoch_XXX.h5)
├── models/           # Final saved model
└── reports/          # All visualizations and metrics
    ├── confusion_matrix.png
    ├── class_performance.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── training_comparison.png
    ├── accuracy_curves.png
    ├── loss_curves.png
    ├── training_history.png
    ├── learning_rate.png
    ├── loss_distribution.png
    ├── prediction_confidence.png
    ├── class_metrics.csv
    └── training_log.json
```

## GPU Memory Optimization

This implementation includes several optimizations to prevent OOM errors during long training runs:

- `TF_GPU_ALLOCATOR=cuda_malloc_async` - Prevents memory fragmentation
- GPU memory growth - Allocates memory as needed
- Memory cleanup callback - Clears session cache between epochs
- Reduced batch size (32) and image size (224x224) for stable training

## License

MIT
