# Real-Time Market Sentiment Analyzer  
*(LangChain + Google Gemini + yfinance + MLflow)*

## Overview
This project implements a simple, notebook friendly pipeline that:

1. Accepts a **company name**  
2. Resolves the **stock ticker** via **Yahoo Finance HTTP** lookup (with a tiny static fallback)  
3. Fetches **latest news** for that ticker using **yfinance**  
4. Analyzes the news with **Gemini 2.0 Flash** via **LangChain** using the exact chain:
   chain1 = prompt_temp | model | parser
5. Logs params, spans, raw news, and the final structured JSON to **MLflow**

Tool-calling is driven by `generate_response(...)` loop, which triggers:
- `resolve_ticker(company)`  
- `fetch_news_yf(symbol, k)`

Deterministic values are then retrieved by calling the same Python functions directly and fed into the LangChain chain for structured sentiment output.

---

## Setup

### Python & Packages:
        pip install --upgrade pip
        pip install yfinance requests mlflow
        pip install langchain langchain-core
        pip install langchain-google-genai
        pip install google-genai

### Configuration:
1. MLflow Tracking URI & Experiment
      ```python
        mlflow.set_tracking_uri("http://20.75.92.162:5000/")
        mlflow.set_experiment("Vignesh_Assignment_1")
        Output link - http://20.75.92.162:5000/#/experiments/348087188207124926/runs/45f0f0ed0e1b4147bf1bdce4bc690120
      ```
        

2. Model & Chain
      ```python
          model_name = "gemini-2.0-flash"
          model = init_chat_model(model_name, model_provider="google_genai")
          parser = StrOutputParser()
          chain1 = prompt_temp | model | parser 
      ```

## How to Run ? 
        out = run_pipeline("Tesla", k=10)  # or "Alphabet Inc", "Apple Inc", etc.
        print(json.dumps(out, indent=2))

## What Each Part Does ?

1. `resolve_ticker(company)`  
   Uses Yahoo Finance HTTP `/v1/finance/search` to find the ticker; falls back to a small `STATIC_TICKERS` map.

2. `fetch_news_yf(symbol, k)`  
   Pulls recent headlines via `yfinance.Ticker(symbol).news`.

3. `format_headlines_block(items)`  
   Safely formats/normalizes mixed field types from yfinance into a readable block for the LLM (handles integers/dicts, epoch timestamps, etc.).

4. **Tool calling** – `generate_response(prompt)`  
   Drives automatic function calls for:  
   `resolve_ticker` → returns ticker  
   `fetch_news_yf` → returns recent news  
   `resolve_ticker` → returns ticker  
   `fetch_news_yf` → returns recent news

5. `utilize_tools(company_name, k)`  
   Triggers the tool calls (side-effect) and then retrieves deterministic outputs by calling the same Python functions directly.  
   Returns: `{"stock_code": ..., "news_items": ...}`.

6. `run_pipeline(company_name, k)`  
   Starts an MLflow run  
   Calls `utilize_tools`  
   Builds `headlines = format_headlines_block(news_items)`  
   Executes the LangChain chain `prompt_temp | model | parser`  
   Parses the LLM output into JSON  
   Logs params, spans, and artifacts to MLflow  
   Returns the final structured JSON


## Sample Command to Run the Chain:
            out = run_pipeline("Alphabet Inc", k=10)
            print(json.dumps(out, indent=2))

## Sample Output JSON (for Tesla"):
```json
[
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Elon Musk's wealth fluctuates; Oracle CEO Larry Ellison briefly surpasses him.",
    "sentiment": "Neutral",
    "people_names": [
      "Ramzan Karmali",
      "Elon Musk",
      "Larry Ellison"
    ],
    "places_names": [],
    "other_companies_referred": [
      "Oracle"
    ],
    "related_industries": [],
    "market_implications": "Highlights volatility in personal wealth linked to stock performance.",
    "confidence_score": 0.7
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Tesla stock experiences a breakout, driven by factors beyond EVs.",
    "sentiment": "Positive",
    "people_names": [],
    "places_names": [],
    "other_companies_referred": [],
    "related_industries": [
      "EV"
    ],
    "market_implications": "Suggests strong upward momentum for Tesla stock.",
    "confidence_score": 0.8
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Wall Street giants have conflicting views on stock market drivers.",
    "sentiment": "Neutral",
    "people_names": [],
    "places_names": [
      "Wall Street"
    ],
    "other_companies_referred": [],
    "related_industries": [],
    "market_implications": "Indicates uncertainty and differing opinions on market trends.",
    "confidence_score": 0.5
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Wolfe Research is optimistic about Tesla's energy business due to recent innovations.",
    "sentiment": "Positive",
    "people_names": [],
    "places_names": [],
    "other_companies_referred": [],
    "related_industries": [
      "Energy"
    ],
    "market_implications": "Suggests potential growth and investment opportunities in Tesla's energy sector.",
    "confidence_score": 0.8
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Tesla is considered a top EV stock with record growth and successful Robotaxi launch.",
    "sentiment": "Positive",
    "people_names": [],
    "places_names": [
      "Austin"
    ],
    "other_companies_referred": [],
    "related_industries": [
      "EV",
      "Robotaxis"
    ],
    "market_implications": "Indicates strong performance and innovation in the EV and autonomous vehicle markets.",
    "confidence_score": 0.9
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Tesla has a higher short interest compared to other Mag 7 stocks.",
    "sentiment": "Negative",
    "people_names": [],
    "places_names": [],
    "other_companies_referred": [],
    "related_industries": [],
    "market_implications": "Suggests potential bearish sentiment and vulnerability to short squeezes.",
    "confidence_score": 0.7
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Renewable energy opportunities in the Middle East and Africa.",
    "sentiment": "Neutral",
    "people_names": [],
    "places_names": [
      "Middle East",
      "Africa",
      "Algeria",
      "Egypt",
      "South Africa",
      "UAE"
    ],
    "other_companies_referred": [],
    "related_industries": [
      "Renewable Energy"
    ],
    "market_implications": "Highlights potential for growth in renewable energy markets.",
    "confidence_score": 0.6
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Tesla is talked about on Wallstreetbets.",
    "sentiment": "Neutral",
    "people_names": [],
    "places_names": [],
    "other_companies_referred": [],
    "related_industries": [],
    "market_implications": "Tesla is a popular topic on retail investor forums.",
    "confidence_score": 0.5
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Shanghai Ant Lingbo Technology Co. showcases humanoid robot, competing with Tesla.",
    "sentiment": "Neutral",
    "people_names": [],
    "places_names": [
      "Shanghai"
    ],
    "other_companies_referred": [
      "Unitree Robotics"
    ],
    "related_industries": [
      "Robotics"
    ],
    "market_implications": "Indicates increasing competition in the humanoid robot market.",
    "confidence_score": 0.7
  },
  {
    "company_name": "Tesla",
    "stock_code": "TSLA",
    "newsdesc": "Larry Ellison's net worth increases, but he remains behind Elon Musk.",
    "sentiment": "Neutral",
    "people_names": [
      "Larry Ellison",
      "Elon Musk"
    ],
    "places_names": [],
    "other_companies_referred": [],
    "related_industries": [],
    "market_implications": "Highlights the competitive landscape of wealth among corporate leaders.",
    "confidence_score": 0.7
  }
]
```
## Notes / Tips

1. If a ticker can’t be resolved, add entries to the STATIC_TICKERS map in resolve_ticker.

2. yfinance news payloads can vary; format_headlines_block already normalizes fields to avoid .strip()-type errors.
