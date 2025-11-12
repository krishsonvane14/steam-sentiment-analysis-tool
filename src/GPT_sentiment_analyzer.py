import os
import json
import time

from dotenv import load_dotenv
from openai import OpenAI

# load local environment
load_dotenv()

# apply api key and create open ai client
openai_api_key = os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)

def analyze_sentiment(review_text, model="gpt-5"):
    """
    send a string of text to GPT to classify sentiment as positive or negative
    :param review_text: text to classify
    :param model: GPT model to use, defaults to gpt5
    :return: striped text response Positive or Negative
    """

    response = openai_client.responses.create(
        model = model,
        input = f"Classify this game review from Steam as Positive or Negative: '{review_text}'"
    )

    return response.output_text.strip();

def batch_classify_from_jsonl(input_jsonl, output_path=None, limit=None, api_call_delay=0.3):

    # load the reviews
    with open(input_jsonl, "r", encoding="utf-8") as reviews_file:
        for index, line in enumerate(reviews_file):
            if limit and index >= limit:
                break

            review = json.loads(line)
            review_text = review.get("review","")
            voted_up = review.get("voted_up","")
            title = review.get("title","")

            review_sentiment = analyze_sentiment(review_text);

            print(f"[{index+1}] GPT-Sentiment: {review_sentiment}, Positive Review: {voted_up}")
            time.sleep(api_call_delay)

    return

batch_classify_from_jsonl("data/reviews.jsonl",limit=10)