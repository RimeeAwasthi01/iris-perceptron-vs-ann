# Iris Flower Classification: Perceptron vs Neural Network

A beginner-friendly comparison of two classifiers on the classic **Iris dataset**:

1. A single-layer **Perceptron** (scikit-learn)
2. A small **multi-layer neural network** (TensorFlow / Keras)

## Dataset
[Iris dataset](https://www.kaggle.com/datasets/uciml/iris) – 150 samples, 3 balanced classes
(*Iris-setosa, Iris-versicolor, Iris-virginica*), 4 features: sepal length/width and petal length/width (cm).

## Workflow
1. Load data and run basic EDA (`head`, `info`, class counts, pairplot)
2. Drop `Id`, encode labels with `LabelEncoder`
3. Stratified 80/20 train-test split
4. Standardise features with `StandardScaler` (fit on train only)
5. Train a Perceptron baseline
6. Train a Keras model: `Dense(16, relu) -> Dense(8, relu) -> Dense(3, softmax)`
   (Adam, categorical cross-entropy, 100 epochs, batch size 8, 20% validation split)
7. Evaluate and plot training vs validation accuracy

## Results
| Model | Test accuracy |
|---|---|
| Perceptron | ~86.7% |
| Keras ANN | ~93.3% |

*(Exact numbers vary slightly with the corrected scaling step; see notebook.)*

## Key takeaways
- The Perceptron is a linear model, so it struggles to separate versicolor and virginica, which overlap.
- A small hidden-layer network captures this non-linearity and performs better.
- With only 30 test samples, differences of one or two predictions move accuracy a lot; cross-validation would give a more reliable estimate.

## Getting started
```bash
git clone https://github.com/RimeeAwasthi01/iris-perceptron-vs-ann.git
cd iris-perceptron-vs-ann
pip install -r requirements.txt
jupyter notebook iris_classification.ipynb
```
Place `Iris.csv` in the same folder as the notebook (or in `data/` and update the path).

## Tech stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras

## Author
Rimee Awasthi

