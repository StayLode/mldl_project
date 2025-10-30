# Tiny ImageNet Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) for image classification on the **Tiny ImageNet** dataset using **PyTorch**.  
It includes dataset preparation, model definition, training, validation, and model checkpointing.

---

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/StayLode/mldl_project.git
cd mldl_project
```

### 2. Install dependencies
Install all required Python packages using:
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The project uses the **Tiny ImageNet** dataset, which contains 200 classes, each with 500 training images and 50 validation images.

To download and extract the dataset, run the following commands (for example, in Google Colab):
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d tiny-imagenet
```

After extraction, the dataset will be located at:
```
tiny-imagenet/tiny-imagenet-200/
```

---

## Project Structure

```
├── checkpoints/           # Saved model checkpoints (.pth)
├── data/                  # Dataset setup and DataLoader scripts
│   └── dataset_setup.py
├── dataset/               # Actual dataset files (Tiny ImageNet)
├── models/                # Model architectures (PyTorch modules)
│   └── custom_net.py
├── utils/                 # Utility functions (training, visualization, etc.)
│   ├── train_utils.py
│   └── visualization.py
├── train.py               # Training pipeline
├── eval.py                # Evaluation script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## Training

To start the training process, run:
```bash
python3 train.py
```

This script:
- Loads and preprocesses the dataset
- Trains the CNN for a specified number of epochs
- Evaluates the model on the validation set
- Saves model checkpoints to the `checkpoints/` directory

The best validation accuracy achieved during training is printed at the end.

---

## Evaluation

To evaluate a trained model, run:
```bash
python3 eval.py
```

This script loads the most recent checkpoint from `checkpoints/` and computes classification accuracy on the validation set.

---

## Notes

- Training can be performed on GPU if available.  
- The dataset and checkpoints are not included in the repository due to size constraints.  
- For reproducibility, random seeds can be set manually in `train.py`.

---

## Requirements

The project requires the following main dependencies (included in `requirements.txt`):

- Python ≥ 3.9  
- torch ≥ 2.0.0  
- torchvision ≥ 0.15.0  
- numpy ≥ 1.24.0  
- matplotlib ≥ 3.7.0  
- tqdm ≥ 4.65.0  
- Pillow ≥ 9.4.0  

---

## License

This project is distributed for educational and research purposes.