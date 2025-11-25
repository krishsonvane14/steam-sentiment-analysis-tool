from cProfile import label
from numpy import full
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import sys
import joblib
import matplotlib.pyplot as plt
import csv

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def logistical_regression(csv_filepath):
    print("Loading CSV")
    full_df = pd.read_csv(csv_filepath)

    required_cols = ["cleaned_review", "sentiment"]
    for col in required_cols:
        if col not in full_df.columns:
            raise ValueError(f"Column '{col}' not found in CSV.")

    print(f"Loaded {len(full_df)} records.\n")
    print("Sentiment distribution:")
    print(full_df["sentiment"].value_counts(), "\n")

    # Encode labels
    full_df["encoded_label"] = LabelEncoder().fit_transform(full_df["sentiment"])
    full_df["cleaned_review"] = full_df["cleaned_review"].astype(str)

    # Balance dataset
    pos_df = full_df[full_df["sentiment"] == "positive"].sample(24000, random_state=42)
    neg_df = full_df[full_df["sentiment"] == "negative"].sample(24000, random_state=42)
    full_df = pd.concat([pos_df, neg_df], ignore_index=True)

    print("Using balanced dataset:")
    print(full_df["sentiment"].value_counts(), "\n")

    # 70% train, 20% val, 10% test
    x_train, x_temp, y_train, y_temp = train_test_split(
        full_df["cleaned_review"],
        full_df["encoded_label"],
        test_size=0.3,
        stratify=full_df["encoded_label"],
        random_state=42
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=1/3,
        stratify=y_temp,
        random_state=42
    )

    print("Vectorizing text\n")
    vectorizer = TfidfVectorizer(max_features=30000)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    # Convert to tensors
    x_train = torch.tensor(x_train_vec.toarray()).float()
    x_val = torch.tensor(x_val_vec.toarray()).float()
    x_test = torch.tensor(x_test_vec.toarray()).float()

    y_train = torch.tensor(y_train.values).float().view(-1, 1)
    y_val = torch.tensor(y_val.values).float().view(-1, 1)
    y_test = torch.tensor(y_test.values).float().view(-1, 1)

    # Model setup
    model = LogisticRegressionModel(x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00075, weight_decay=1e-5)

    print("Training model\n")
    epochs = 20
    training_loss = []
    validation_loss = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            
        training_loss.append(loss.item())
        validation_loss.append(val_loss.item())

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {loss.item():.4f} - "
            f"Val Loss: {val_loss.item():.4f}"
        )

    print("\nTraining Complete\n")

    # Final test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)

        probs = torch.sigmoid(test_outputs)
        preds = (probs > 0.5).float()

    accuracy = (preds.eq(y_test)).float().mean().item() 

    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    y_true_np = y_test.cpu().numpy().astype(int)
    y_pred_np = preds.cpu().numpy().astype(int)

    TP = int(((y_pred_np == 1) & (y_true_np == 1)).sum())
    TN = int(((y_pred_np == 0) & (y_true_np == 0)).sum())
    FP = int(((y_pred_np == 1) & (y_true_np == 0)).sum())
    FN = int(((y_pred_np == 0) & (y_true_np == 1)).sum())

    cm = [[TN, FP],
      [FN, TP]]

    # Save model and vectorizer
    torch.save(model.state_dict(), "logistical_model.pth")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    
    # Plot 
    plt.figure(figsize=(6, 5))
    plt.plot(training_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Model Training Curve")
    plt.legend()
    plt.grid(True)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Label positions
    plt.xticks([0, 1], ["Negative", "Positive"])
    plt.yticks([0, 1], ["Negative", "Positive"])

    # Add cell values
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)

    plt.colorbar()

    # Save confusion matrix
    cm_plot_path = "logistic_confusion_matrix.png"
    plt.savefig(cm_plot_path)
    plt.close()

    # Save to file
    plot_path = "logistic_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    # Save training info to CSV
    metrics = {
        "model_type": "logistic_regression",
        "features": x_train.shape[1],
        "epochs": epochs,
        "learning_rate": 0.001,
        "test_loss": float(test_loss.item()),
        "test_accuracy": float(accuracy),
        "true_negative": TN,
        "false_positive": FP,
        "false_negative": FN,
        "true_positive": TP,
    }

    csv_path = "logistic_model_metrics.csv"
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
        print("Usage: python3 logistic_2.py <reviews_cleaned.csv>")
    else:
        csv_filepath = sys.argv[1]
        logistical_regression(csv_filepath)
