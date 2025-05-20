# Federated Animal Classifier

Federated Learning-based binary classification of animal images, simulated across 1000 clients. This project implements a privacy-preserving decentralized machine learning system that trains a CNN model without centralized data collection.

## 📌 Project Overview

This project explores the theoretical foundations and practical implementation of Federated Learning (FL) in the context of image classification. Specifically, it focuses on a binary classification task (e.g., Cat vs. Dog) using the PetImages dataset and simulates the participation of 1000 clients in a federated setup.

**Key Objectives:**
- Understand and apply core FL principles such as local training and federated averaging.
- Simulate large-scale federated training using image data.
- Evaluate the performance of a CNN in a federated setting.

## 🧠 Key Concepts

- **Federated Learning (FL):** A decentralized learning technique that enables edge devices (clients) to collaboratively train a shared global model without exposing raw data.
- **Privacy Preservation:** Local datasets remain on client devices, and only model updates are shared with the central server.
- **Model Aggregation:** Updates from all clients are combined using Federated Averaging (FedAvg) to form the global model.

## 🏗️ Folder Structure
FederatedAnimalClassifier/
│
├── main.py # Entry point: Contains simulation logic
├── PetImages/
│ ├── labels.csv # CSV file with image labels (binary)
│ └── train/
│ ├── <image1>.jpg
│ ├── <image2>.jpg
│ └── ... # All images used for training
└── README.md # This file

## 🧪 Dataset

- **Source:** Kaggle's [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)
- **Preprocessing:** 
  - Images resized and normalized
  - Labels formatted into a binary `labels.csv` file
  - Distributed virtually across simulated clients

## ⚙️ How it Works

1. **Initialization:** A global CNN model is initialized and broadcasted to all clients.
2. **Local Training:** Each client trains the model on its local data subset.
3. **Update Aggregation:** Clients send model updates (not raw data) back to the server.
4. **Global Model Update:** The central server aggregates updates using FedAvg.
5. **Iteration:** Steps 2-4 repeat for multiple federated learning rounds.

## 📊 Model Architecture

A Convolutional Neural Network (CNN) with:
- Convolution + ReLU layers
- MaxPooling
- Fully Connected Dense layers
- Binary classification output (Sigmoid)

## 🔁 Simulation

- **Number of Clients:** 1000 (simulated in parallel/sequential batches)
- **Client Sampling:** Random subset selected per round (default: 10%)
- **Framework Used:** PyTorch

## 📈 Results

The model achieved promising performance while ensuring data privacy. Accuracy improved steadily over multiple communication rounds, highlighting the effectiveness of FL even in non-IID and distributed data conditions.

## 🔧 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install dependencies with:
```bash
pip install -r requirements.txt
```
## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/<your-username>/federated-animal-classifier.git
cd federated-animal-classifier
```
2. Run the simulation:
```bash
python main.py
```

## Contributors
- Kunj Mehul Doshi
- Vaibhav Gupta
- Aman Rameshchandra Agrawal
---
This project was developed as part of a study on the theoretical foundations and practical implementation of Federated Learning at BITS Pilani.
---
