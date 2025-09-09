import streamlit as st
from sentiment_stock_predictor import main
from visualization import create_all_visualizations

st.title("Sentiment-Driven Stock Price Predictor")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
if st.button("Run Prediction"):
    predictor, merged_df, news_df, report, predicted_next_price = main(ticker)
    st.success("Prediction complete!")
    st.write(f"Predicted next day's closing price for {ticker}: **{predicted_next_price:.2f}**")
    create_all_visualizations(merged_df, news_df, None, predictor.performance_metrics, report)
    st.image("visualizations/sentiment_distribution.png")
    st.image("visualizations/model_comparison.png")
    st.image("visualizations/performance_summary.png")