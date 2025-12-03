# steam-sentiment-analysis-tool

Sentiment analysis tool for Steam game reviews. Fetches real user reviews using the Steam API and generates an objective aggregate sentiment score for any game, providing buyers with a more transparent, data-driven view than store ratings.

---

# Main Linear Model

## Training Model

To run the main model

```
cd src/
python ./train_model.py <input_csv> -<flag>
```

Arguments

- `<input_csv>`: Path to the cleaned CSV file containing the reviews (e.g., ./data/cleaned_reviews.csv)
- `<flag>`: Choose a flag
  - `-t`: train the model from scratch.
  - `-l`: load the weights from training.

---

# Web App

## Features

- Preprocessing and TF-IDF vectorisation
- PyTorch linear classifier
- Web interface for querying any Steam game
- Backend FastAPI for predictions
- Visaulize model metrics and prediction summaries

## Installation

1. Clone the Repository

```
git clone https://github.com/krishsonvane14/steam-sentiment-analysis-tool.git
cd steam-sentiment-analysis-tool
```

2. Install Dependencies

```
pip install -r requirements.txt
```

## Running the Web App

```
cd src/app
python ./start.py
```

This script launches the FastAPI backend and opens the frontend interface in your defaul browser.
If it does not open, visit:

```
http://127.0.0.1:5500/index.html
```

- Troubleshooting:
  - The table contents may take a second to load
  - Refresh a few times to fix it

## Model Information

The saved model weights are located at:

```
/src/app/backend/linear_model.pth
/src/app/backend/logistical_model.pth
```

The saved vectorizers are located at:

```
/src/app/backend/linear_tfidf_vectorizer.pkl
/src/app/backend/logistic_tfidf_vectorizer.pkl
```

To retrain the models in the backend

```
cd src/app/backend
python ./linear_model.py <input.csv>
python ./logistic_model.py <input.csv>
```

## Example Outputs

<div align="center">

<img src="src/app/backend/linear_loss_plot.png" alt="Linear Loss Plot" width="500"/>

<img src="src/app/backend/linear_accuracy_plot.png" alt="Linear Accuracy Plot" width="500"/>

<img src="src/app/backend/linear_confusion_matrix.png" alt="Linear Confusion Matrix" width="500"/>

</div>


# GPT Sentiment Analyzer
## Setup
Because using any of the GPT models in an application context requires interfacing directly with OpenAI’s APIs, there is some setup required before any analysis can be done. The user must first create an OpenAI account. This gives access to OpenAI’s public AI tools such as ChatGPT and Sora but also to the OpenAI Platform which is the main developer portal for those wishing to utilize OpenAI’s services. 

OpenAI, unlike some other LLM developers, does not provide a free tier of access to any of its GPT models, Access is billed based on input and output tokens, the model being queried and priority of access. It is necessary to create a billing plan on the OpenAI platform, adding a payment method and paying for credits to be added. Next, in the account API keys manager a new valid API key must be generated, with the private key saved locally. For security the private key is never included in the program code or entered by the user during runtime, instead is stored in a local `.env` file where the program can access it as an environment variable. A template `.env.example` is included to fill in.

## Running the Program

GPT Sentiment Analyzer can be run from the terminal emulator by navigating to the directory containing the script and entering `python GPT_sentiment_analyzer` the program has commands for its three modes of operation.

`classify [filepath].csv` targets a csv with reviews’ text in a particular column and call the OpenAI response API to create a copy of the dataset  appended with additional columns for the text of the sentiment the model determined (negative or positive), the normalized numeric sentiment (0 or 1). There are five optional arguments for this command:

* `--reviews_col [column]`: specifies the name of the column of reviews to classify, defaults to `“cleaned_review”`
`reviews_sentiment_col [column]`: specifies the name of the column with the reviewer assigned numeric sentiment, not technically required for classification but needed for later evaluation, defaults to “encoded_senti”
* `model [model name]`: the GPT model used for the classification, tested with `“gpt-5”`, `“gpt-5-mini”`, and `“gpt-5-nano”`, use of `“gpt-5-nano"` is recommended and this is the default
* `limit [positive integer]`: classify up to the specified number of reviews in the dataset, defaults to none
* `–o [newfilepath]`: save the csv with classifications and API info to the specified filepath, defaults to `[filepath]_[model]_sentiment.csv`


`Evaluate [filepath]`: provides metrics and statistics on predicted sentiment classification of a dataset against actual sentiment. In the console it will display the confusion matrix, the classification report for the confusion matrix, and a statistics report on the API’s performance. These results are saved to descriptively named and timestamped files. If the dataset csv contains rows with undefined sentiment, these are dropped from the evaluation and included in a separate csv. There are three optional arguments:
* `--by [column]`: splits the evaluation of rows to be grouped by the unique items in the specific `[column]`, useful if per genre or per game results are desired.  
* `--actual_sentiment_col [column]`: 
* `--predicted_sentiment_col [column]`:

`Sample [filepath].csv`: randomly selects rows of a csv of reviews, useful if the existing dataset is large and classification of the entire set is infeasible or undesirable. There are three optional arguments:
* `--size [positive integer]`: How many reviews to include in the sample, default is 100
* `--random [positive integer]`: random seed for selection, default is 42
* `--stratify [column]`: stratifies the sample so that for each unique label in `[column]` similar number of rows for each label will be included, no default
