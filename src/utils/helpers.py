def clean_text(text):
    # Function to clean the input text by removing unwanted characters and formatting
    cleaned_text = text.strip().replace('\n', ' ').replace('\r', '')
    return cleaned_text

def format_date(date_string):
    # Function to format date strings into a standard format
    from datetime import datetime
    formatted_date = datetime.strptime(date_string, '%Y-%m-%d').date()
    return formatted_date

def extract_ticker(symbol):
    # Function to extract stock ticker from a string
    return symbol.upper() if symbol else None

# Additional utility functions can be added here as needed.