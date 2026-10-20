# CIFAR-10 Robustness Probe

How much does a small CNN's accuracy depend on clean labels and clean pixels?
This project trains a compact convolutional network on CIFAR-10 and then probes
it with three controlled perturbations: fully shuffled labels, 20% label noise,
and additive Gaussian input noise.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Results

The baseline CNN is trained for 30 epochs, then evaluated on the held-out test
set and on three perturbed copies of its training split. All numbers are read
from the confusion matrices in [`results/`](results/).

| Evaluation set | Samples | Accuracy | What it shows |
|---|---:|---:|---|
| Baseline (test set) | 10,000 | **72.3%** | Generalisation of the clean model |
| Random label shuffle | 40,000 | 10.3% | Chance level, as expected: labels carry no signal |
| 20% label noise | 40,000 | 76.5% | Close to the ~80% ceiling set by the corrupted labels |
| Gaussian input noise (σ = 0.1) | 40,000 | 43.8%\* | See note below |

\* The recorded input-noise run clamped noisy images to `[0, 1]` even though
inputs are normalised to `[-1, 1]`, which erased every negative pixel value. Most
of that drop comes from the clamp, not the noise. The bug is fixed in
`src/main.py` (it now clamps to `[-1, 1]`); re-run the script to get the corrected
figure.

<p align="center">
  <img src="results/Baseline_Test_confusion_matrix.png" width="560" alt="Baseline test-set confusion matrix">
</p>

The baseline confusion matrix shows the familiar CIFAR-10 pattern: vehicles
(ship, truck, automobile) are separated well, while the animal classes, especially
cat↔dog and bird↔deer, absorb most of the errors.

## Model

`SimpleCNN` in [`src/main.py`](src/main.py):

```
Conv(3→32) → ReLU → MaxPool
Conv(32→64) → ReLU → MaxPool
Conv(64→128) → ReLU → MaxPool
FC(2048→256) → ReLU → FC(256→128) → ReLU → FC(128→10)
```

Adam (lr = 1e-3), batch size 128, 30 epochs, 80/20 train/validation split,
seed 42.

## Running it

```bash
pip install -r requirements.txt
python src/main.py
```

- CIFAR-10 downloads automatically to `data/`.
- The trained weights are cached at `models/baseline_model.pth`; delete the file to retrain.
- Confusion matrices and the loss curve are written to `results/`.

## Project structure

```
├── src/
│   └── main.py          # training, perturbation experiments, evaluation
├── results/             # confusion matrices and loss curve
├── requirements.txt
└── README.md
```

## Notes

- The perturbation experiments evaluate the already-trained baseline on modified
  copies of its training split; they measure robustness of a fixed model, not the
  effect of training on noisy data.
- Built as a mini-project for an AI course at Seoul National University (2025).

## License

[MIT](LICENSE) © Dongkun Moon
