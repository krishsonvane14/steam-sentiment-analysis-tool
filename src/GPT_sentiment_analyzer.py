import json
import os
import time

import pandas as pd
import sklearn as skl
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Literal

from spyder.plugins.help.utils.sphinxify import usage

# load local environment
load_dotenv()

# apply api key and create open AI client
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)

# define structured response for batch review processing
class SentimentInput(BaseModel):
    prompt: str
    reviews: List[str]

sentiment_input_schema = SentimentInput.model_json_schema()

class SentimentResponse(BaseModel):
    sentiments: List[Literal["positive","negative"]]

sentiment_response_schema = SentimentResponse.model_json_schema()


def single_analyze_sentiment(client, review_text, model="gpt-5-mini", max_retries = 5, retry_sleep = 1, prompt_content = None, sentiment_labels=None):

    if sentiment_labels is None:
        sentiment_labels = ["positive", "negative"]
    if prompt_content is None:
        prompt_content = "Classify the sentiment of this video game review from Steam as 'positive' or 'negative': "

    prompt = f"{prompt_content}{review_text}"

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model = model,
                input =  prompt
            )

            # normalize and validate response
            response_normalized = response.output_text.strip().lower()
            if response_normalized not in sentiment_labels:
                raise ValueError(f"unrecognized sentiment label: {response_normalized}")

            # return sentiment and usage stats as dict
            return {
                "sentiment": response_normalized,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }

        except:
            if attempt + 1 == max_retries:
                raise RuntimeError(f"failed after {max_retries} retries")
            time.sleep(retry_sleep * (attempt + 1))

    return None


def classify_from_jsonl(input_jsonl, output_path=None, limit=None, api_call_delay=0.3):

    # load the reviews
    with open(input_jsonl, "r", encoding="utf-8") as reviews_file:
        for index, line in enumerate(reviews_file):
            if limit and index >= limit:
                break

            review = json.loads(line)
            review_text = review.get("review","")
            voted_up = review.get("voted_up","")
            title = review.get("title","")

            review_sentiment = single_analyze_sentiment(review_text)

            print(f"[{index+1}] GPT-Sentiment: {review_sentiment}, Positive Review: {voted_up}")
            time.sleep(api_call_delay)

    return



def batch_analyze_sentiment(reviews_batch, gpt_model="gpt-5-mini", max_retries = 5, retry_sleep = 2, prompt_content=None):

    # default content
    if prompt_content is None:
        prompt_content = (
            "Label each of following video game reviews from Steam as 'positive' or 'negative'."
        )

    input_structure = SentimentInput(prompt=prompt_content,reviews=reviews_batch)

    for attempt in range(max_retries):
        try:
            response = openai_client.responses.parse(
                model = gpt_model,
                input = input_structure.model_dump(),
                text_format = sentiment_response_schema
            )

            # return response.output[0].parsed.sentiments
            return response.output_parsed

        except Exception as e:
            if attempt + 1 == max_retries:
                raise
            time.sleep(retry_sleep * (attempt + 1))

    return None

def classify_reviews_from_csv(client,
                              input_csv,
                              gpt_model = "gpt-5-mini",
                              output_csv=None,
                              review_col="cleaned_review",
                              gpt_sentiment_col = "gpt_sentiment",
                              input_tokens_col = "input_tokens",
                              output_tokens_col = "output_tokens",
                              reviews_limit=None,
                              progress_interval = 20):

    reviews_dataframe = pd.read_csv(input_csv)

    # head off dataframe if limit is present
    if reviews_limit is not None:
        reviews_dataframe = reviews_dataframe.head(reviews_limit)


    reviews_count = len(reviews_dataframe)
    print(f"Analyzing Sentiment of {reviews_count} reviews from {input_csv} using {gpt_model}")

    # add new columns to dataframe
    reviews_dataframe[gpt_sentiment_col] = None
    reviews_dataframe[input_tokens_col] = 0
    reviews_dataframe[output_tokens_col] = 0

    analysis_start_time = time.perf_counter()

    for index, review_text in enumerate(reviews_dataframe[review_col].astype(str)):

        try:
            sentiment_result = single_analyze_sentiment(client, review_text, model=gpt_model)

            reviews_dataframe.loc[index, gpt_sentiment_col] = sentiment_result["sentiment"]
            reviews_dataframe.loc[index, input_tokens_col] = sentiment_result["input_tokens"]
            reviews_dataframe.loc[index, output_tokens_col] = sentiment_result["output_tokens"]

        except Exception as e:
            print(f"[ERROR] row {index}: {e}")
            reviews_dataframe.loc[index, gpt_sentiment_col] = "error"

        if (index + 1) % progress_interval == 0 or index + 1 == reviews_count:
            print(f"showing review {index + 1}/{reviews_count} analysis:")
            print(reviews_dataframe.iloc[index, 1:])

    # print finish message with stats
    analysis_end_time = time.perf_counter()
    analysis_total_time = analysis_end_time - analysis_start_time
    total_token_input = reviews_dataframe[input_tokens_col].sum()
    total_token_output = reviews_dataframe[output_tokens_col].sum()

    print(f"\ncompleted in {analysis_total_time:.3f} seconds, total {total_token_input} tokens inputted and {total_token_output} tokens outputted")

    # procedurally generate title for output csv if no title is provided
    if output_csv is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = f"{root}_{gpt_model}_sentiment{ext}"

    # save file
    reviews_dataframe.to_csv(output_csv,index=False)
    print(f"saved with appended sentiment to {output_csv}")

    return reviews_dataframe, output_csv, analysis_total_time


def evaluate_sentiment_classifer_from_csv(input_csv, actual_sentiment_col="sentiment", predicted_sentiment_col="gpt_sentiment",
                                          sentiment_labels=None, output_csv=None):
    # get the confusion matrix
    if sentiment_labels is None:
        sentiment_labels = ["positive", "negative"]
    classified_dataframe = pd.read_csv(input_csv)

    actual_sentiments = classified_dataframe[actual_sentiment_col].tolist()
    predicted_sentiments = classified_dataframe[predicted_sentiment_col].tolist()

    sentiment_confusion_matrix = skl.metrics.confusion_matrix(actual_sentiments, predicted_sentiments, labels=sentiment_labels)
    sentiment_report = skl.metrics.classification_report(actual_sentiments, predicted_sentiments, labels=sentiment_labels,output_dict=True)

    sentiment_report_df = pd.DataFrame(sentiment_report).transpose()

    # print confusion matrix and classification report
    print(f"Confusion Matrix for {input_csv}")
    print(sentiment_confusion_matrix)

    print(f"\nClassification Report for {input_csv}")
    print(sentiment_report_df.to_string(float_format="%.3f"))

    # save classification report
    if output_csv is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = root + "_report" + ext

    sentiment_report_df.to_csv(output_csv, index=True)
    print(f"\nreport saved to {output_csv}")

    return sentiment_confusion_matrix, output_csv


def generate_reviews_sample_csv(input_csv, size = 100, random_state=49,stratified_by=None, output_csv=None):
    reviews_dataframe = pd.read_csv(input_csv)

    sample, _ = skl.model_selection.train_test_split(
        reviews_dataframe,
        train_size=size,
        random_state=random_state,
        stratify=reviews_dataframe[stratified_by] if stratified_by else None)

    if output_csv is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = root + "_sample_" + "rows_" + str(size) + "_state_" + str(random_state)

        if stratified_by:
            output_csv += "_stratified_" + stratified_by

        output_csv += ext

    sample.to_csv(output_csv, index=False)

    return output_csv

fake_reviews = ["Great story and music.", "This game crashes constantly.","Pretty good but too short.", "Horrible, do not play ever!"]