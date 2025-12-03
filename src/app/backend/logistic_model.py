import time
import sys

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from torch import nn, optim

import matplotlib.pyplot as plt
import joblib

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def logistical_regression(csv_filepath):
    print("Loading CSV")

    full_df = pd.read_csv(csv_filepath)
    full_df['encoded_senti'] = LabelEncoder().fit_transform(full_df['sentiment'])
    
    seen_df = full_df.loc[(full_df['__appid'] != 377160) & (full_df['__appid'] != 4000)]

    unseen_1 = full_df.loc[full_df['__appid'] == 377160]
    unseen_2 = full_df.loc[full_df['__appid'] == 4000]
    unseen_df = pd.concat([unseen_1, unseen_2])
    
    # print(unseen_df)

    # export the test set csv for the LLM
    # unseen_df.to_csv('test_reviews.csv', index=False)
    
    # splitting the dataset up
    neg_df = seen_df.loc[seen_df['encoded_senti'] == 0]
    pos_df = seen_df.loc[seen_df['encoded_senti'] == 1]
    pos_df = pos_df.sample(n=neg_df.shape[0], random_state=42)

    '''
    neg_df = full_df.loc[full_df['encoded_senti'] == 0]
    pos_df = full_df.loc[full_df['encoded_senti'] == 1]
    pos_df = pos_df.sample(n=neg_df.shape[0], random_state=42)
    '''
    np_df = [neg_df, pos_df]
    train_df = pd.concat(np_df)

    # TRAINING PREPARATION
    # split into train and test sets with random seed 42
    
    x_train, x_val, y_train, y_val = train_test_split(
        train_df['cleaned_review'], train_df['encoded_senti'], test_size=.1, stratify=train_df['sentiment'],
        random_state=42
    )

    _, x_test, _, y_test = train_test_split(
        unseen_df['cleaned_review'], unseen_df['encoded_senti'], test_size=12836, stratify=unseen_df['sentiment'],
        random_state=42
    )
   
    print("Vectorizing text\n")

    # convert x_train and x_test to vectors to tensors
    vectorizer = TfidfVectorizer(max_features=3000)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)

    x_train = torch.tensor(x_train.toarray()).float()
    x_test = torch.tensor(x_test.toarray()).float()
    x_val = torch.tensor(x_val.toarray()).float()

    y_train = torch.tensor(y_train.values).float().view(-1, 1)
    y_val = torch.tensor(y_val.values).float().view(-1, 1)
    y_test = torch.tensor(y_test.values).float().view(-1, 1)

    # Model setup
    model = LogisticRegressionModel(x_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00075, weight_decay=1e-5)

    print("Training model\n")
    epochs = 20
    train_losses = []
    val_losses = []
    val_acc= []
    
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()

        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        train_loss = loss.item()
        train_losses.append(train_loss)
        optimizer.step()

        # Validate
        with torch.no_grad():
            model.eval()
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
            
            probs = torch.sigmoid(val_outputs)
            preds = (probs > 0.5).float()
            val_accuracy = (preds.eq(y_val)).float().mean().item()
            val_acc.append(val_accuracy)

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {loss.item():.4f} - "
            f"Val Loss: {val_loss.item():.4f} - "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    print("time to train and eval: ", str(time.time() - start_time))
    print("\nTraining Complete\n")

    # Save model and vectorizer
    torch.save(model.state_dict(), "logistic_model.pth")
    joblib.dump(vectorizer, "logistic_tfidf_vectorizer.pkl")
    
    # Confusion matrix
    with torch.no_grad():
        model.eval()
        val_outputs = model(x_val)
        probs = torch.sigmoid(val_outputs)
        preds = (probs > 0.5).float()
    
    cm = confusion_matrix(
        y_val.cpu().numpy().ravel(), 
        preds.cpu().numpy().ravel()
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])

    plt.figure(figsize=(6,5))
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix")
    plt.grid(False)
    plt.savefig('logistic_confusion_matrix.png')
    plt.close()

    # Final test evaluation
    with torch.no_grad():
        model.eval()
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
        test_probs = torch.sigmoid(test_outputs)
        test_preds = (test_probs > 0.5).float()
        test_acc = (test_preds.eq(y_test)).float().mean().item()
        
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
          
    # Plot 
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Logistic Model Training and Validation Loss")
    plt.legend()
    plt.savefig("logistic_loss_plot.png")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 4))
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Logistic Model Validation Accuracy per Epoch")
    plt.legend()
    plt.savefig("logistic_accuracy_plot.png")
    plt.close()

    # Save training info to CSV
    metrics = {
        "model_type": "logistic_regression",
        "features": x_train.shape[1],
        "epochs": epochs,
        "learning_rate": 0.00075,
        "train_loss": float(train_losses[-1]),
        "validation_loss": float(val_losses[-1]),
        "validation_accuracy": float(val_acc[-1]),
        "test_accuracy": float(test_acc)
    }

    pd.DataFrame([metrics]).to_csv("logistic_metrics.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 logistic_2.py <reviews_cleaned.csv>")
    else:
        csv_filepath = sys.argv[1]
        logistical_regression(csv_filepath)
