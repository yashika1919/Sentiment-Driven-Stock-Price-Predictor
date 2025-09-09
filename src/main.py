# main.py

import os
from src.data.loader import load_data
from src.sentiment.analyzer import analyze_sentiment
from src.model.predictor import train_model, predict_stock_prices

def main():
    # Load data
    news_data, stock_data = load_data()

    # Analyze sentiment
    sentiment_scores = analyze_sentiment(news_data)

    # Train model
    model = train_model(stock_data, sentiment_scores)

    # Predict stock prices
    predictions = predict_stock_prices(model, stock_data)

    # Output predictions
    print(predictions)

if __name__ == "__main__":
    main()