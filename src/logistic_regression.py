import jsonlines
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
import sys
import matplotlib.pyplot as plt

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # w^T x + b
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        # sigma(w^T x + b)
        return torch.sigmoid(self.linear(x))


def logistical_regression(json_filepath):
    print("Starting Training")
    full_df = pd.DataFrame()
    count = 0

    # Load data 
    with jsonlines.open(json_filepath) as reader:
        # stop = number of reviews
        stop = 1000
        for obj in reader:
            if count == (stop - 1):
                break

            df = pd.DataFrame(obj)
            df = df.drop([
                'steamid', 'num_games_owned', 'num_reviews',
                'playtime_forever', 'playtime_last_two_weeks', 'last_played'
            ])
            df = df[['review', 'voted_up']]
            full_df = pd.concat([full_df, df], ignore_index=True)
            count += 1

    print(f"Loaded {len(full_df)} reviews.\n")

    # Encode labels 
    full_df['encoded_vote'] = LabelEncoder().fit_transform(full_df['voted_up'])

    # Clean text 
    full_df['review'] = (
        full_df['review']
        .str.lower()
        .str.replace("[h1]", " ", regex=False)
        .str.replace("[/h1]", " ", regex=False)
        .str.replace("[h2]", " ", regex=False)
        .str.replace("[/h2]", " ", regex=False)
        .str.replace("[b]", " ", regex=False)
        .str.replace("[/b]", " ", regex=False)
    )

    # Split for training and testing
    x_train, x_test, y_train, y_test = train_test_split(
        full_df['review'], full_df['encoded_vote'],
        test_size=0.2, stratify=full_df['voted_up'], random_state=42
    )

    print("Vectorizing text\n")
    # max_features = number of unique words
    vectorizer = TfidfVectorizer(max_features=5000)
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
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()

    # lr = learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    test_accuracies = []

    print("Training model\n")
    epochs = 10
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
        
        train_losses.append((loss.item()))

        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

            # Binary predictions
            preds = (test_outputs > 0.5).float()
            test_accuracy = (preds.eq(y_test)).float().mean().item()
            test_accuracies.append(test_accuracy)
    
        print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {loss.item():.4f} - "
          f"Test Loss: {test_loss.item():.4f} - "
          f"Test Accuracy: {test_accuracy*100:.2f}%")

    print("Training Complete")

    # Plot Losses 
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.show()
    
    # Plot Test Accuracy 
    plt.figure(figsize=(10,4))
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy per Epoch')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 logistic_regression.py <input.jsonl>")
    else:
        json_filepath = sys.argv[1]
        logistical_regression(json_filepath)