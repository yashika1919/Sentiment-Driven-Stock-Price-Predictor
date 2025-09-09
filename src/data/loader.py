def load_data(file_path):
    """
    Load data from a specified file path.
    
    Args:
        file_path (str): The path to the data file.
        
    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    import pandas as pd
    
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """
    Preprocess the loaded data for analysis.
    
    Args:
        data (DataFrame): The raw data to preprocess.
        
    Returns:
        DataFrame: A pandas DataFrame containing the preprocessed data.
    """
    # Example preprocessing steps
    data.dropna(inplace=True)  # Remove missing values
    data['date'] = pd.to_datetime(data['date'])  # Convert date column to datetime
    return data


def load_financial_news(news_file_path):
    """
    Load financial news articles from a specified file path.
    
    Args:
        news_file_path (str): The path to the news articles file.
        
    Returns:
        DataFrame: A pandas DataFrame containing the loaded news articles.
    """
    return load_data(news_file_path)


def load_stock_data(stock_file_path):
    """
    Load stock price data from a specified file path.
    
    Args:
        stock_file_path (str): The path to the stock price data file.
        
    Returns:
        DataFrame: A pandas DataFrame containing the loaded stock price data.
    """
    return load_data(stock_file_path)