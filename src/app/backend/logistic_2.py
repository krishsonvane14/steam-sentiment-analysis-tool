import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import sys
import joblib


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # w^T x + b
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # sigma(w^T x + b)
        # return torch.sigmoid(self.linear(x))
        return self.linear(x)


def logistical_regression(csv_filepath):
        print("Loading CSV")
        full_df = pd.read_csv(csv_filepath)
        
        required_cols = ["cleaned_review", "sentiment"]
        for col in required_cols:
            if col not in full_df.columns:
                raise ValueError(f"Column '{col}' not found in CSV.")
        
        print(f"Loaded {len(full_df)} records.\n")

        print("Sentiment distribution: ")
        print(full_df["sentiment"].value_counts(), "\n")

        # Encode labels
        full_df["encoded_label"] = LabelEncoder().fit_transform(full_df["sentiment"])

        # Use pre-cleaned text
        full_df["cleaned_review"] = full_df["cleaned_review"].astype(str)

        # Uncommnet to use balance dataset 
        """ 
        pos_df = full_df[full_df["sentiment"] == "positive"].sample(24000, random_state=42)
        neg_df = full_df[full_df["sentiment"] == "negative"].sample(24000, random_state=42)

        full_df = pd.concat([pos_df, neg_df], ignore_index=True)
        print("Using balanced dataset:")
        print(full_df["sentiment"].value_counts())
        """

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            full_df["cleaned_review"],
            full_df["encoded_label"],
            test_size=0.2,
            stratify=full_df["sentiment"],
            random_state=42
        )

        print("Vectorizing text\n")
        # max_features = number of unique words
        vectorizer = TfidfVectorizer(max_features=30000)
        x_train = vectorizer.fit_transform(x_train)
        x_test = vectorizer.transform(x_test)

        # Convert to torch tensors
        x_train = torch.tensor(x_train.toarray()).float()
        x_test = torch.tensor(x_test.toarray()).float()
        y_train = torch.tensor(y_train.values).float().view(-1, 1)
        y_test = torch.tensor(y_test.values).float().view(-1, 1)

        # Logistical regression
        input_dim = x_train.shape[1]
        model = LogisticRegressionModel(input_dim)
        # criterion = nn.BCELoss()
        # criterion = nn.MSELoss()
        criterion = nn.BCEWithLogitsLoss()
        
        # lr = learning rate
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        test_losses = []
        test_accuracies = []

        print("Training model\n")
        epochs = 20
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
            train_losses.append(loss.item())

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(x_test)
                test_loss = criterion(test_outputs, y_test)
                test_losses.append(test_loss.item())

                temp = 0.5
                prob = torch.sigmoid(test_outputs/temp)
                preds = (prob > 0.5).float()

                test_accuracy = (preds.eq(y_test)).float().mean().item()
                test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {loss.item():.4f} - "
                  f"Test Loss: {test_loss.item():.4f} - "
                  f"Test Accuracy: {test_accuracy*100:.2f}%")

        print("\nTraining Complete")
        torch.save(model.state_dict(), "logistical_model.pth")
        joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 logistic_2.py <reviews_cleaned.csv>")
    else:
        csv_filepath = sys.argv[1]
        logistical_regression(csv_filepath)