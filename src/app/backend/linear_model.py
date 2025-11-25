import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import time
import sys
import joblib
import matplotlib.pyplot as plt
import csv


class Sequential(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.lin1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(64, 2)
        self.lsm = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.lsm(x)
        return x

def train_model(filepath):
    print("starting operations\n")

    full_df = pd.read_csv(filepath)
    full_df['encoded_senti'] = LabelEncoder().fit_transform(full_df['sentiment'])
    full_df["cleaned_review"] = full_df["cleaned_review"].astype(str)

    pos_df = full_df[full_df["sentiment"] == "positive"].sample(24000, random_state=42)
    neg_df = full_df[full_df["sentiment"] == "negative"].sample(24000, random_state=42)
    full_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # TRAINING PREPARATION
    # split into train and test sets with random seed 42
    
    x_train, x_temp, y_train, y_temp = train_test_split(
        full_df["cleaned_review"],
        full_df["encoded_senti"],
        test_size=0.30,
        stratify=full_df["encoded_senti"],
        random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1/3,
        stratify=y_temp,
        random_state=42
    )

    print("vectorizing\n")
    
    # convert x_train and x_test to vectors to tensors
    vectorizer = TfidfVectorizer(max_features=30000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    # Convert to tensors
    x_train_t = torch.tensor(x_train_vec.toarray()).float()
    x_val_t = torch.tensor(x_val_vec.toarray()).float()
    x_test_t = torch.tensor(x_test_vec.toarray()).float()

    y_train_t = torch.tensor(y_train.values).long()
    y_val_t = torch.tensor(y_val.values).long()
    y_test_t = torch.tensor(y_test.values).long()
    
    print(x_test.shape)
    print(y_test.shape)

    # linear regression
    input_dim = x_train_t.shape[1]
    model = Sequential(input_dim)
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00075, weight_decay=1e-4)

    print("training and eval: \n")
    start_time = time.time()

    # training and evaluation
    epochs = 20
    print("\nTraining\n")
    training_loss = []
    validation_loss = []
    validation_acc = []

    for e in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()

        output = model(x_train_t)
        train_loss = criterion(output, y_train_t)
        train_loss.backward()

        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(x_val_t)
            val_loss = criterion(val_output, y_val_t)

            val_preds = torch.argmax(val_output, dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        training_loss.append(train_loss.item())
        validation_loss.append(val_loss.item())
        validation_acc.append(val_acc)

        print(
            f"Epoch {e+1}/{epochs}  "
            f"Train Loss: {train_loss.item():.4f}  "
            f"Val Loss: {val_loss.item():.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

    print("\nTraining complete\n")
    
    # Save model and vectorizer
    torch.save(model.state_dict(), "linear_model.pth")
    joblib.dump(vectorizer, "tfidf_vectorizer_linear.pkl")
    
    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(x_test_t)
        test_loss = criterion(test_output, y_test_t)
        test_preds = torch.argmax(test_output, dim=1)
        test_acc = (test_preds == y_test_t).float().mean().item()

    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    
    # Plot training curve
    plt.figure(figsize=(6, 5))
    plt.plot(training_loss, label="Train Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("linear_plot.png")
    plt.close()

    # Confusion matrix
    y_true = y_test_t.numpy()
    y_pred = test_preds.numpy()

    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())

    cm = [[TN, FP],
          [FN, TP]]

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)

    plt.colorbar()
    plt.savefig("linear_confusion_matrix.png")
    plt.close()
    
    # Save training info to CSV
    metrics = {
        "model_type": "linear",
        "features": x_train_t.shape[1],
        "epochs": epochs,
        "learning_rate": 0.00075,
        "test_loss": float(test_loss.item()),
        "test_accuracy": float(test_acc),
        "true_negative": TN,
        "false_positive": FP,
        "false_negative": FN,
        "true_positive": TP,
    }

    csv_path = "linear_model_metrics.csv"
    file_exists = False
    try:
        with open(csv_path, "r"):
            file_exists = True
    except FileNotFoundError:
        file_exists = False

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <input>")
    else:
        filepath = sys.argv[1]
        train_model(filepath)
        