# Neural Networks for Image and Text Classification

This repository brings together two deep-learning studies: an adaptive convolutional network for image classification and a TextCNN-based investigation of data efficiency and input representation in text classification.

## Adaptive Multi-Branch CNN — CIFAR-10

The image-classification model is built from adaptive intermediate blocks. Each block applies multiple convolutional branches to the same input and learns input-dependent branch weights from channel-wise summary features before combining the branch outputs.

The architecture was developed through iterative experiments with:

- wider and deeper convolutional blocks
- Batch Normalisation
- Softmax-based branch weighting
- Dropout regularisation
- MaxPooling
- data augmentation
- label smoothing
- SGD with momentum and Nesterov acceleration
- weight decay and cosine learning-rate scheduling

The final enhanced model achieved **92.65% test accuracy on CIFAR-10**.

## TextCNN — AG News

The text-classification study uses **TextCNN** on the four-class **AG News** dataset to examine how model performance changes when labelled training data and input context are reduced.

The experiments compare:

- randomly initialised embeddings
- **GloVe** embeddings
- **FastText** embeddings
- training subsets from **1% to 100%** of the available training pool
- full-text and short-context inputs
- different word-selection strategies, including positional, random, TF-IDF-based and salience-based selection

The short-context experiments use fixed word budgets to study how much information can be removed while preserving classification performance. RoBERTa is used only as a word-salience selector in those experiments; **TextCNN remains the classifier**.

Evaluation includes **accuracy, Macro-F1, per-class F1, confusion matrices, confidence intervals, training time and inference time**.

## Tech

**Python · PyTorch · scikit-learn · pandas · NumPy · Matplotlib · GloVe · FastText**

## Repository Contents

- `Group_AJ_ECS7026P4_Coursework_Part_1.ipynb` — adaptive CNN experiments on CIFAR-10
- `Zobeiry_TextCNN_AGNews_Combined_Final(1).ipynb` — TextCNN experiments on AG News
