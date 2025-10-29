
import jsonlines
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch import optim
import scipy
import numpy
import matplotlib
import matplotlib.pyplot as plt

import sys


def train_model(json_filepath):
    print("starting operations\n")
    full_df = pd.DataFrame()
    count = 0
    # convert jsonl to dataframe objects to store in full_df
    with jsonlines.open(json_filepath) as reader:
        
        stop = 200
        for obj in reader:
            
            if count == stop:
                break;
            
            df = pd.DataFrame(obj)
            # drop other rows
            df = df.drop(['steamid','num_games_owned','num_reviews','playtime_forever','playtime_last_two_weeks','last_played'])
            # drop columns
            #df = df.drop(['recommendationid', 'author', 'language', 'timestamp_created', 'timestamp_updated', 'votes_funny',
            #              'comment_count', 'primarily_steam_deck'], axis=1)
            df = df[['review', 'voted_up']]

            full_df = pd.concat([full_df, df], ignore_index=True)
            count = count + 1

        print("finished appending \n")
        full_df['encoded_vote'] = LabelEncoder().fit_transform(full_df['voted_up'])

    # clean up the review text for the vectorizer
    full_df['review'] = full_df['review'].str.lower()
    full_df['review'] = full_df['review'].str.replace("[h1]", ' ').str.replace("[/h1]", ' ').str.replace(
                        "[h2]", ' ').str.replace("[/h2]", ' ').str.replace(
                        "[b]", ' ').str.replace("[/b]", ' ')
    
    
    '''
    # check what the review text looks like
    print(full_df['review'].str.split().apply(len).describe())
    print("review one\n")
    print(full_df[full_df['review'].str.split().apply(len) < 90]['review'].values[0])
    
    print("review two\n")
    print(full_df[full_df['review'].str.split().apply(len) > 130]['review'].values[0][:500])
    '''

    # TRAINING PREPARATION
    # split into train and test sets with random seed 42
    x_train, x_test, y_train, y_test = train_test_split(
        full_df['review'], full_df['encoded_vote'], test_size=.2, stratify=full_df['voted_up'],
        random_state=42
    )
    #print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print("\n")
    print("vectorizing\n")
    
    # convert x_train and x_test to vectors to tensors
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    x_train = torch.tensor(scipy.sparse.csr_matrix.todense(x_train)).float()
    x_test = torch.tensor(scipy.sparse.csr_matrix.todense(x_test)).float()

    y_train = torch.tensor(y_train.values)
    y_test = torch.tensor(y_test.values)

    # linear regression
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 2),
        nn.LogSoftmax(dim=1)
    )

    criterion = nn.NLLLoss()

    fPass = model(x_train)
    loss = criterion(fPass, y_train)

    loss.backward()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []
    test_accuracies = []

    # training and evaluation
    epochs = 30
    for e in range(epochs):
        optimizer.zero_grad()

        output = model.forward(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        train_loss = loss.item()
        train_losses.append(train_loss)

        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            fPass = model(x_test)
            test_loss = criterion(fPass, y_test)
            test_losses.append(test_loss)

            newPass = torch.exp(fPass)
            top_p, top_class = newPass.topk(1, dim=1)
            equals = top_class == y_test.view(*top_class.shape)
            test_accuracy = torch.mean(equals.float())
            test_accuracies.append(test_accuracy)

        model.train()

        print(f"Epoch: {e+1}/{epochs}.. ",
            f"Training Loss: {train_loss:.3f}.. ",
            f"Test Loss: {test_loss:.3f}.. ",
            f"Test Accuracy: {test_accuracy:.3f}")
        



    plt.figure(figsize=(12, 5))
    ax = plt.subplot(121)
    plt.xlabel('epochs')
    plt.ylabel('negative log likelihood loss')
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.show()
    plt.legend(frameon=False);
    plt.subplot(122)
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.plot(test_accuracies)
    plt.show()
    # FOR FINAL TOOL:
    # look towards cross validation
    # how much review data we need to consider
    # include confusion matrix
    # time the training sessions

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_model.py <input>")
    else:
        json_filepath = sys.argv[1]
        train_model(json_filepath)
