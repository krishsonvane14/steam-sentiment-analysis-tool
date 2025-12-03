# steam-sentiment-analysis-tool
Sentiment analysis tool for Steam game reviews. Fetches real user reviews using the Steam API and generates an objective aggregate sentiment score for any game, providing buyers with a more transparent, data-driven view than store ratings. 

---

# Web App

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
