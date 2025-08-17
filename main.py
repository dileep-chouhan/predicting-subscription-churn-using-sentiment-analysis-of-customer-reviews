import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Download VADER lexicon if not already present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_reviews = 200
data = {
    'CustomerID': np.arange(1, num_reviews + 1),
    'Review': [f'Review {i}: This is a sample review text. ' + ('Positive' if i % 2 == 0 else 'Negative') for i in range(num_reviews)],
    'Churned': np.random.choice([True, False], size=num_reviews, p=[0.2, 0.8]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
analyzer = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Review'].apply(lambda review: analyzer.polarity_scores(review)['compound'])
df['SentimentCategory'] = df['Sentiment'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral'))
# --- 3. Data Analysis ---
churn_by_sentiment = df.groupby('SentimentCategory')['Churned'].mean()
print("Churn Rate by Sentiment Category:")
print(churn_by_sentiment)
# --- 4. Visualization ---
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_sentiment.index, y=churn_by_sentiment.values)
plt.title('Churn Rate vs. Review Sentiment')
plt.xlabel('Sentiment Category')
plt.ylabel('Churn Rate')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'churn_sentiment.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 5.  Further Analysis (Optional):  Could incorporate more sophisticated modeling here.  Example below: ---
# This section is commented out to keep the example concise but demonstrates further analysis possibilities.
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# # Prepare data for model training
# X = df[['Sentiment']]
# y = df['Churned']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # Train a logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# # Make predictions
# y_pred = model.predict(X_test)
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy}")