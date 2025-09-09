class SentimentAnalyzer:
    def __init__(self):
        from textblob import TextBlob
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.textblob_analyzer = TextBlob
        self.vader_analyzer = SentimentIntensityAnalyzer()

    def analyze_textblob(self, text):
        analysis = self.textblob_analyzer(text)
        return analysis.sentiment.polarity

    def analyze_vader(self, text):
        score = self.vader_analyzer.polarity_scores(text)
        return score['compound']

    def analyze_sentiment(self, text):
        tb_score = self.analyze_textblob(text)
        vader_score = self.analyze_vader(text)
        return {
            'textblob_score': tb_score,
            'vader_score': vader_score
        }