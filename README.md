# Ratings Classification with Neural Network and Logistic Regression

This repository contains a Python implementation for a ratings classification task using two approaches:
- A **Feedforward Neural Network** built with [PyTorch](https://pytorch.org/).
- A **Logistic Regression** model using [scikit-learn](https://scikit-learn.org/).

The code loads an Excel file (`Ratings_exercise.xlsx`), preprocesses the data, and then trains and evaluates both models. It demonstrates:
- Data preprocessing using Pandas and StandardScaler.
- Model training with K-Fold cross validation.
- Evaluation with accuracy scores, classification reports, and confusion matrices.
- A baseline comparison between the neural network and logistic regression.

---

## Features

- **Data Preprocessing:**  
  Reads the Excel file and scales the features.
  
- **Neural Network Model:**  
  Implements a feedforward neural network with one hidden layer using PyTorch.

- **K-Fold Cross Validation:**  
  Trains and evaluates the model on 5 different folds.

- **Full Dataset Training:**  
  Provides an option to train on the entire dataset without cross-validation.

- **Baseline Model:**  
  Compares the neural network with a logistic regression model from scikit-learn.

---

## Requirements

- Python 3.7+
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [openpyxl](https://pypi.org/project/openpyxl/) (for reading Excel files)

### Installation

Install the required packages using pip:

```bash
pip install pandas numpy torch scikit-learn openpyxl
```

---

## Project Structure

- **Data Loading and Preprocessing:**  
  Loads `Ratings_exercise.xlsx`, extracts specific columns as features, and maps rating values to numeric labels.

- **Model Definition:**  
  Defines a `FeedforwardNN` class for the neural network with a hidden layer and ReLU activation.

- **Training & Evaluation (With K-Folds):**  
  Uses 5-fold cross validation to train the model and evaluates performance on each fold.

- **Training & Evaluation (No Folds):**  
  Trains the model on the entire dataset and evaluates its performance.

- **Baseline Logistic Regression:**  
  Implements a logistic regression model to compare with the neural network's performance.

---

## How to Run

1. **Prepare Your Data:**  
   Ensure the Excel file `Ratings_exercise.xlsx` is in your project directory and contains the necessary columns:
   - `rel_size`, `excess_rets`, `idio_stdev`, `ni_ta`, `tl_ta`
   - `ratings9` (target variable)

2. **Run the Notebook or Script:**  
   - If using a Jupyter Notebook, open it and run all cells.
   - Alternatively, save the code in a Python script (e.g., `ratings_classification.py`) and run it:

   ```bash
   python ratings_classification.py
   ```

3. **Review the Output:**  
   The script prints out the accuracy, classification reports, and confusion matrices for both the neural network and logistic regression models for each fold and for the full dataset.

---

## Code Example

Below is a complete example of the code:

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
data = pd.read_excel('Ratings_exercise.xlsx')

# Preprocess features and labels
features = data[['rel_size', 'excess_rets', 'idio_stdev', 'ni_ta', 'tl_ta']]
labels = data['ratings9'].astype(int)  

unique_ratings = sorted(labels.unique())
rating_to_idx = {r: i for i, r in enumerate(unique_ratings)}
labels = labels.map(rating_to_idx)
num_classes = len(unique_ratings)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = features_scaled
y = labels.values

# Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_dim = X.shape[1]      # number of predictors (5)
hidden_dim = 10000          # hidden layer size (adjustable)
output_dim = num_classes    # number of classes
num_epochs = 100            # training epochs
learning_rate = 0.001       # learning rate

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    print(f"Fold {fold}")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Initialize model, loss function, and optimizer
    model = FeedforwardNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluation for this fold
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, predicted = torch.max(test_outputs, 1)

    nn_acc = accuracy_score(y_test, predicted.numpy())
    print(f"Neural Network Accuracy: {nn_acc:.4f}")
    print("Neural Network Classification Report:")
    print(classification_report(y_test, predicted.numpy()))
    print("Neural Network Confusion Matrix:")
    print(confusion_matrix(y_test, predicted.numpy()))
    
    # Logistic Regression Baseline
    logit_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logit_model.fit(X_train, y_train)
    logit_pred = logit_model.predict(X_test)
    logit_acc = accuracy_score(y_test, logit_pred)
    print(f"Logistic Regression Accuracy: {logit_acc:.4f}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, logit_pred))
    print("Logistic Regression Confusion Matrix:")
    print(confusion_matrix(y_test, logit_pred))
    print("-" * 50)
    
    fold_results.append({'nn_acc': nn_acc, 'logit_acc': logit_acc})

nn_avg_acc = np.mean([res['nn_acc'] for res in fold_results])
logit_avg_acc = np.mean([res['logit_acc'] for res in fold_results])
print(f"Average Neural Network Accuracy: {nn_avg_acc:.4f}")
print(f"Average Logistic Regression Accuracy: {logit_avg_acc:.4f}")

# Training and evaluation on the full dataset (No K-Folds)
data = pd.read_excel('Ratings_exercise.xlsx')

features = data[['rel_size', 'excess_rets', 'idio_stdev', 'ni_ta', 'tl_ta']]
labels = data['ratings9'].astype(int)

unique_ratings = sorted(labels.unique())
rating_to_idx = {r: i for i, r in enumerate(unique_ratings)}
labels = labels.map(rating_to_idx)
num_classes = len(unique_ratings)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X = features_scaled
y = labels.values

# Define the neural network again (same as above)
class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_dim = X.shape[1]      
hidden_dim = 10000 
output_dim = num_classes    
num_epochs = 100            
learning_rate = 0.001       

X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

model = FeedforwardNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

nn_acc = accuracy_score(y, predicted.numpy())
print("Neural Network Accuracy: {:.4f}".format(nn_acc))
print("Neural Network Classification Report:")
print(classification_report(y, predicted.numpy(), target_names=[str(r) for r in unique_ratings]))
print("Neural Network Confusion Matrix:")
print(confusion_matrix(y, predicted.numpy()))

# Logistic Regression on full dataset
logit_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
logit_model.fit(X, y)
logit_pred = logit_model.predict(X)
logit_acc = accuracy_score(y, logit_pred)
print("\nLogistic Regression Accuracy: {:.4f}".format(logit_acc))
print("Logistic Regression Classification Report:")
print(classification_report(y, logit_pred, target_names=[str(r) for r in unique_ratings]))
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y, logit_pred))
```

---

## Notes

- **Hyperparameter Tuning:**  
  The `hidden_dim`, `num_epochs`, and `learning_rate` parameters can be adjusted to improve model performance.

- **Data File:**  
  Ensure that `Ratings_exercise.xlsx` is in the correct format and available in your working directory.

- **Environment:**  
  The code is designed to be run in a Jupyter Notebook, but it can also be executed as a standalone Python script.

---

## License

This project is licensed under the GNU License. 

---

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)

---

Simply copy and paste the contents of this README into your GitHub repository's `README.md` file to get started!
