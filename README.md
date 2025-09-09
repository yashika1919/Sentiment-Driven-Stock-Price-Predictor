# Sentiment-Driven Stock Price Predictor

## Overview
The Sentiment-Driven Stock Price Predictor is a Python project that leverages sentiment analysis of financial news articles to predict stock prices. The project utilizes various libraries for data processing, sentiment analysis, and machine learning to provide insights into stock market trends.

## Project Structure
```
Sentiment-Driven-Stock-Price-Predictor
├── src
│   ├── main.py
│   ├── data
│   │   └── loader.py
│   ├── sentiment
│   │   └── analyzer.py
│   ├── model
│   │   └── predictor.py
│   └── utils
│       └── helpers.py
├── requirements.txt
├── .vscode
│   └── settings.json
├── README.md
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd Sentiment-Driven-Stock-Price-Predictor
   ```

2. **Create a Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

4. **VS Code Configuration**
   Ensure that your VS Code is set to use the virtual environment's Python interpreter. This can be done by adding the following to `.vscode/settings.json`:
   ```json
   {
       "python.pythonPath": "venv/bin/python"
   }
   ```

## Running the Project
To run the project, follow these steps:

1. **Run the Main Script**
   Execute the main script to start the workflow:
   ```bash
   python src/main.py
   ```

2. **Train the Model**
   After analyzing the sentiment, run the predictor script to train the model:
   ```bash
   python src/model/predictor.py
   ```

3. **Visualize Results**
   Finally, visualize the results using the appropriate scripts in the `visualizations` directory (if applicable).

## Troubleshooting Common Issues
- **ModuleNotFoundError**: Ensure that your virtual environment is activated.
- **Package Installation Issues**: Check your internet connection and try reinstalling the packages.
- **VS Code Not Recognizing Virtual Environment**: Restart VS Code after activating the environment.

## Contribution
Feel free to contribute to this project by submitting issues or pull requests. Your feedback and contributions are welcome!

## License
This project is licensed under the MIT License.