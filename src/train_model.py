import jsonlines
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
from torch import nn
from torch import optim
import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt
import time
import sys


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

    seen_df = full_df.loc[full_df['__appid'] != 377160]
    seen_df = seen_df.loc[full_df['__appid'] != 4000]

    unseen_1 = full_df.loc[full_df['__appid'] == 377160]
    unseen_2 = full_df.loc[full_df['__appid'] == 4000]
    unseen_df = pd.concat([unseen_1, unseen_2])
    print(unseen_df)
    # export the test set csv for the LLM
    unseen_df.to_csv('test_reviews.csv', index=False)


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

    '''
    alternative of train-val-test split
    make sure that seen data includes the unseen data
    x_train, x_val, y_train, y_val = train_test_split(
        train_df['cleaned_review'], train_df['encoded_senti'], 
        test_size=.2, stratify=train_df['sentiment'], random_state=42
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=1./8, stratify=train_df['sentiment'],
        random_state=42
    )

    '''
    print("vectorizing \n")
    
    
    # convert x_train and x_test to vectors to tensors
    vectorizer = TfidfVectorizer(max_features=3000)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)

    x_train = torch.tensor(x_train.toarray()).float()
    x_test = torch.tensor(x_test.toarray()).float()
    x_val = torch.tensor(x_val.toarray()).float()

    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)
    y_val = torch.tensor(y_val.values)

    # linear regression
    input_dim = x_train.shape[1]
    model = Sequential(input_dim)
    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.00075)

    train_losses = []
    val_losses = []
    val_accuracies = []
    pred_label = []

    print("training and eval: \n")
    start_time = time.time()

    # training and evaluation
    epochs = 20
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model.forward(x_train)
        loss = criterion(output, y_train)

        loss.backward()
        train_loss = loss.item()
        train_losses.append(train_loss)

        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            fPass = model(x_val)
            val_loss = criterion(fPass, y_val)
            val_losses.append(val_loss)

            newPass = torch.exp(fPass)
            top_p, top_class = newPass.topk(1, dim=1)
            equals = top_class == y_val.view(*top_class.shape)
            val_accuracy = torch.mean(equals.float())
            val_accuracies.append(val_accuracy)

        print(f"Epoch: {e+1}/{epochs}.. ",
            f"Training Loss: {train_loss:.3f}.. ",
            f"Val. Loss: {val_loss:.3f}.. ",
            f"Val. Accuracy: {val_accuracy:.3f}")
    
    print("time to train and eval: ", str(time.time() - start_time))
    
    # confusion matrix
    with torch.no_grad():
        model.eval()
        fPass = model(x_val)
        #val_loss = criterion(fPass, y_val)
        #test_losses.append(test_loss)

        newPass = torch.exp(fPass)
        _, top_class = newPass.topk(1, dim=1)
        pred_label = top_class
        equals = top_class == y_val.view(*top_class.shape)
        val_accuracy = torch.mean(equals.float())
        val_accuracies.append(val_accuracy)

    cm = confusion_matrix(y_val, pred_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix")
    plt.grid(False)
    plt.savefig('linreg_fig.png')

    # test model with unseen data
    model.eval()
    output = model(x_test)
    #print(output)
    output = torch.exp(output)
    #print(newPass)
    _, top_class = output.topk(1, dim=1)
    pred_label = top_class
    equals = top_class == y_test.view(*top_class.shape)
    test_accuracy = torch.mean(equals.float())
    print("accuracy: ", test_accuracy.item()*100, "%")


    # Plot Losses 
    plt.figure(figsize=(10,4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    
    # Plot Test Accuracy 
    plt.figure(figsize=(10,4))
    plt.plot(val_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy per Epoch')
    plt.legend()
    plt.savefig('test_plot.png')
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <input>")
    else:
        filepath = sys.argv[1]
        train_model(filepath)
