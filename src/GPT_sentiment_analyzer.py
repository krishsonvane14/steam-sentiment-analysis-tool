import argparse
import os
import time
import typing

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as skl
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

matplotlib.use('Agg') # generate figures without GUI

class SentimentResult(typing.NamedTuple):
    sentiment: str
    input_tokens_count: int
    output_tokens_count: int
    response_time: float

# load local environment
load_dotenv()

# apply api key and create open AI client
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

def single_analyze_sentiment(client, review_text, model="gpt-5-nano", max_retries = 5, retry_sleep = 1, prompt_content = None, sentiment_labels=None):

    if sentiment_labels is None:
        sentiment_labels = ["positive", "negative"]
    if prompt_content is None:
        prompt_content = "Classify the sentiment of this video game review from Steam as 'positive' or 'negative': "

    prompt = f"{prompt_content}{review_text}"

    # attempt calling openAI
    for attempt in range(max_retries):
        try:

            api_call_start_time = time.perf_counter()

            response = client.responses.create(
                model = model,
                input =  prompt
            )

            api_call_end_time = time.perf_counter()

            # normalize and validate response
            response_normalized = response.output_text.strip().lower()
            if response_normalized not in sentiment_labels:
                raise ValueError(f"unrecognized sentiment label: {response_normalized}")

            # return sentiment and usage stats
            return SentimentResult(
                sentiment = response_normalized,
                input_tokens_count= response.usage.input_tokens,
                output_tokens_count= response.usage.output_tokens,
                response_time= api_call_end_time - api_call_start_time
            )

        except Exception as e:
            print(f"attempt {attempt+1} failed: {e}")
            if attempt + 1 == max_retries:
                raise RuntimeError(f"failed after {max_retries} retries")
            time.sleep(retry_sleep * (attempt + 1))

    return None

def normalize_binary_label(label):
    """
    converts a binary label to 1 for a positive result and 0 for a negative result
    """

    if label is None:
        raise ValueError("label cannot be None")

    label = str(label).strip().lower()

    # positive label case
    if label in ["positive", "pos", "true", "t"]:
        return 1

    # negative label case:
    if label in ["negative", "neg", "false", "f"]:
        return 0

    raise ValueError(f"unrecognized label: {label}")

def classify_reviews_from_csv(client,
                              input_csv,
                              gpt_model = "gpt-5-nano",
                              output_csv=None,
                              review_col="cleaned_review",
                              gpt_sentiment_col = "gpt_sentiment",
                              gpt_sentiment_norm_col ="gpt_sentiment_norm",
                              input_tokens_col = "input_tokens",
                              output_tokens_col = "output_tokens",
                              response_time_col = "response_time",
                              reviewer_sentiment_col = "encoded_senti",
                              reviewer_sentiment_norm_col = "sentiment_norm",
                              reviews_limit=None,
                              save_increment = 20,
                              progress_interval = 1):

    # procedurally generate title for output csv if no title is provided
    if output_csv is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = f"{root}_{gpt_model}_sentiment{ext}"

    reviews_dataframe = pd.read_csv(input_csv, encoding="utf-8")

    # check csv for correct columns
    if review_col not in reviews_dataframe.columns:
        raise ValueError(f"Review column {review_col} not in {input_csv}")
    if reviewer_sentiment_col not in reviews_dataframe.columns:
        raise ValueError(f"Review sentiment column {reviewer_sentiment_col} not in {input_csv}")

    # head off dataframe if limit is present
    if reviews_limit is not None:
        reviews_dataframe = reviews_dataframe.head(reviews_limit)


    # add normalized user_reviews column
    # reviewer_sentiment_col_idx = reviews_dataframe.columns.get_loc(reviewer_sentiment_col)
    # reviews_dataframe.insert(reviewer_sentiment_col_idx + 1, reviewer_sentiment_norm_col, reviews_dataframe[reviewer_sentiment_col].apply(normalize_binary_label))

    reviews_count = len(reviews_dataframe)
    print(f"Analyzing Sentiment of {reviews_count} reviews from {input_csv} using {gpt_model}")

    # add new columns to dataframe
    reviews_dataframe[gpt_sentiment_col] = None
    reviews_dataframe[gpt_sentiment_norm_col] = None
    reviews_dataframe[input_tokens_col] = 0
    reviews_dataframe[output_tokens_col] = 0
    reviews_dataframe[response_time_col] = 0.0

    analysis_start_time = time.perf_counter()

    try:
        for index, review_text in enumerate(reviews_dataframe[review_col].astype(str)):

            try:
                sentiment_result = single_analyze_sentiment(client, review_text, model=gpt_model)

                # normalize result
                try:
                    numeric_sentiment = normalize_binary_label(sentiment_result.sentiment)
                except ValueError as e:
                    print(f"[ERROR] row {index}: {e}")
                    numeric_sentiment = None

                reviews_dataframe.loc[index, gpt_sentiment_col] = sentiment_result.sentiment
                reviews_dataframe.loc[index, gpt_sentiment_norm_col] = numeric_sentiment
                reviews_dataframe.loc[index, input_tokens_col] = sentiment_result.input_tokens_count
                reviews_dataframe.loc[index, output_tokens_col] = sentiment_result.output_tokens_count
                reviews_dataframe.loc[index, response_time_col] = sentiment_result.response_time

            except Exception as e:
                print(f"[ERROR] row {index}: {e}")
                reviews_dataframe.loc[index, gpt_sentiment_col] = "error"
                reviews_dataframe.loc[index, gpt_sentiment_norm_col] = None
                reviews_dataframe.loc[index, input_tokens_col] = 0
                reviews_dataframe.loc[index, output_tokens_col] = 0
                reviews_dataframe.loc[index, response_time_col] = 0.0

                continue

            # periodic save
            if (index + 1) % save_increment == 0 or index + 1 == reviews_count:
                reviews_dataframe.to_csv(output_csv, index=False)
                print(f"Checkpoint saved output at row {index + 1}/{reviews_count}")


            if (index + 1) % progress_interval == 0 or index + 1 == reviews_count:
                result_cols = [gpt_sentiment_col, gpt_sentiment_norm_col, input_tokens_col, output_tokens_col, response_time_col]
                review_results = reviews_dataframe.loc[index,result_cols].to_dict()

                print(f"Review {index + 1}/{reviews_count} -> {review_results}")

    # save reviews with sentiment at end of loop or if there is exception
    finally:
        reviews_dataframe.to_csv(output_csv, index=False)
        print(f"final save with appended sentiment to {output_csv}")

    # print finish message with stats
    analysis_end_time = time.perf_counter()
    total_classification_time = analysis_end_time - analysis_start_time
    total_token_input = reviews_dataframe[input_tokens_col].sum()
    total_token_output = reviews_dataframe[output_tokens_col].sum()

    print(f"\ncompleted in {total_classification_time:.3f} seconds, total {total_token_input} tokens inputted and {total_token_output} tokens outputted")

    return output_csv, total_classification_time

BINARY_LABEL_MAPPING = {0:"negative", 1:"positive"}

# helpers for
def plot_save_confusion_matrix(confusion_matrix, label_mapping, title, output_path, colourmap="Greens"):
    display = skl.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                                                 display_labels=list(label_mapping.values()))
    display.plot(cmap=colourmap)
    plt.title(title)
    plt.grid(False)
    plt.savefig(output_path)
    plt.close()


def compute_gpt_response_stats(df, input_tokens_col, output_tokens_col, response_time_col,total_tokens_col):
    stats = ["sum", "mean", "median", "std", "min", "max"]

    # add total tokens column
    df[total_tokens_col] = df[input_tokens_col] + df[output_tokens_col]

    stats_df = df[[response_time_col, input_tokens_col, output_tokens_col, total_tokens_col]].agg(stats)

    return stats_df


def compute_classification_report(actual, predicted, labels):
    report_dict = skl.metrics.classification_report(actual, predicted, labels=labels, output_dict=True)
    return pd.DataFrame.from_dict(report_dict).transpose()


def evaluate_sentiment_classified_csv(csv_to_eval_path,
                                      actual_sentiment_col ="encoded_senti",
                                      predicted_sentiment_col ="gpt_sentiment_norm",
                                      input_tokens_col = "input_tokens",
                                      output_tokens_col = "output_tokens",
                                      response_time_col = "response_time",
                                      group_col=None):

    # load csv and validate
    eval_df = pd.read_csv(csv_to_eval_path)

    required_columns = [actual_sentiment_col, predicted_sentiment_col, input_tokens_col, output_tokens_col,
                        response_time_col]

    for column in required_columns:
        if column not in eval_df.columns:
            raise ValueError(f"column {column} not in {csv_to_eval_path}")

    # remove rows where GPT sentiment is None
    valid_df = eval_df[eval_df[predicted_sentiment_col].notna()].copy()
    failed_df = eval_df[eval_df[predicted_sentiment_col].isna()].copy()

    eval_root, eval_ext = os.path.splitext(csv_to_eval_path)

    # generate timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # report and save failed rows
    if len(failed_df) > 0:
        failed_path = f"{eval_root}_failed_rows_{timestamp}.csv"
        failed_df.to_csv(failed_path, index=False)

        print(f" Dropped {len(failed_df)} of {len(eval_df)} rows in {predicted_sentiment_col} with None sentiment before evaluation\nSaved dropped rows to {failed_path}")

    # dictionary for returned paths
    evaluation_reports_paths = {}

    groups = [None] if group_col is None else eval_df[group_col].unique()

    if group_col is not None:
        print(f"Evaluating sentiment classification in {csv_to_eval_path} by unique categories in {group_col}")
    else:
        print(f"Evaluating sentiment classification in {csv_to_eval_path} for all reviews")

    # loop through groups to generate reports
    for group in groups:
        if group is None:
            subset_df = valid_df
            group_name = "All"
        else:
            subset_df = valid_df[valid_df[group_col] == group]
            group_name = str(group)

        print(f"Generating reports for {group_name}\n")

        # extract sentiment values for comparison
        actual_sentiment_values = subset_df[actual_sentiment_col].tolist()
        predicted_sentiment_values = subset_df[predicted_sentiment_col].tolist()

        # confusion matrix
        confusion_matrix = skl.metrics.confusion_matrix(actual_sentiment_values,predicted_sentiment_values,labels=list(BINARY_LABEL_MAPPING.keys()))
        confusion_matrix_path = f"{eval_root}_{group_name}_confusion_matrix_{timestamp}.png"
        confusion_matrix_title = f"Confusion Matrix for {group_name}"
        plot_save_confusion_matrix(confusion_matrix,BINARY_LABEL_MAPPING,title=confusion_matrix_title,output_path=confusion_matrix_path)
        print(f"Confusion Matrix for {group_name}:\n{confusion_matrix}\n")

        # classification report
        class_report_df = compute_classification_report(actual=actual_sentiment_values, predicted=predicted_sentiment_values, labels=list(BINARY_LABEL_MAPPING.keys()))
        class_report_path = f"{eval_root}_{group_name}_classification_report_{timestamp}.csv"
        class_report_df.to_csv(class_report_path, index=True)
        print(f"Classification report for {group_name}:\n{class_report_df.transpose().to_string(float_format="%.3f")}\n")

        # GPT API stats
        stats_df = compute_gpt_response_stats(subset_df,input_tokens_col,output_tokens_col,response_time_col,"total_tokens")
        stats_path = f"{eval_root}_{group_name}_api_stats_{timestamp}.csv"
        stats_df.to_csv(stats_path, index=True)
        print(f"Statistics report for {group_name}:\n{stats_df.to_string(float_format="%.3f")}\n")

        # add paths to dictionary
        evaluation_reports_paths[group_name] = {
            "confusion_matrix": confusion_matrix_path,
            "classification_report": class_report_path,
            "statistics_report": stats_path,
        }

    return evaluation_reports_paths

def generate_reviews_sample_csv(input_csv, size = 100, random_state=42,stratified_by=None, output_csv=None):
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

def per_app_metrics(classified_df):
    return None

# define the parser for the command line execution
def build_parser():
    parser = argparse.ArgumentParser(description="GPT_sentiment_analyzer CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # classification commands
    classify_parser = subparsers.add_parser("classify", help="classify reviews with GPT model")
    classify_parser.add_argument("csv_to_classify", help="the name of the csv with reviews to classify")
    classify_parser.add_argument("--reviews_col", default="cleaned_review", help="the name of the column in the csv with review text to classify")
    classify_parser.add_argument("--reviewer_sentiment_col", default="encoded_senti")
    classify_parser.add_argument("--model", default="gpt-5-nano")
    classify_parser.add_argument("--limit", type=int, default=None)
    classify_parser.add_argument("--o", default=None)

    # evaluation commands
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate sentiment classification results from csv")
    evaluate_parser.add_argument("csv_to_evaluate", help="the name of the csv with sentiment classification results to evaluate")
    evaluate_parser.add_argument("--actual_sentiment_col", default="encoded_senti")
    evaluate_parser.add_argument("--predicted_sentiment_col", default="gpt_sentiment_norm")
    evaluate_parser.add_argument("--model", default="gpt-5-nano")
    evaluate_parser.add_argument("--by", default=None, help="make separate evaluation reports for each unique item in this column of the dataset")

    # sampling commands
    sampling_parser = subparsers.add_parser("sample", help="sample reviews from a dataset")
    sampling_parser.add_argument("csv_to_sample", help="the name of the csv with reviews to sample")
    sampling_parser.add_argument("--size", type=int, default=100, help="number of reviews to sample")
    sampling_parser.add_argument("--random", type=int, default=42, help="random seed for sampling")
    sampling_parser.add_argument("--stratify", type=str, default=None, help="stratification value")

    return parser

# parse and run commands
def main ():

    args_parser = build_parser()
    args = args_parser.parse_args()

    if args.command == "classify":
        output_csv, _ = classify_reviews_from_csv(client=openai_client,
                                                  input_csv=args.csv_to_classify,
                                                  gpt_model=args.model,
                                                  review_col=args.reviews_col,
                                                  reviewer_sentiment_col=args.reviewer_sentiment_col,
                                                  reviews_limit=args.limit,
                                                  output_csv=args.o
                                                  )
    elif args.command == "evaluate":
        report_paths = evaluate_sentiment_classified_csv(csv_to_eval_path=args.csv_to_evaluate,
                                                         actual_sentiment_col=args.actual_sentiment_col,
                                                         predicted_sentiment_col=args.predicted_sentiment_col,
                                                         group_col=args.by)

    elif args.command == "sample":
        sample_csv = generate_reviews_sample_csv(input_csv=args.csv_to_sample,
                                                 size=args.size,
                                                 random_state=args.random,
                                                 stratified_by=args.stratify)

if __name__ == "__main__":
    main()
