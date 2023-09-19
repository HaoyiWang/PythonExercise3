import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    moby_dick_text = file.read()

words = word_tokenize(moby_dick_text)
filtered_words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]

pos_tags = pos_tag(filtered_words)

fdist = FreqDist(tag for (word, tag) in pos_tags)
common_pos = fdist.most_common(30)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
top_30_tokens = [word for word, _ in FreqDist(filtered_words).most_common(30)]
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in top_30_tokens]

plt.figure(figsize=(12, 6))
x, y = zip(*common_pos)
plt.bar(x, y)
plt.title("Top 30 Parts of Speech")
plt.xlabel("Parts of Speech")
plt.ylabel("Frequency")

analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(sentence)["compound"] for sentence in sent_tokenize(moby_dick_text)]

avg_sentiment = sum(sentiments) / len(sentiments)

overall_sentiment = "positive" if avg_sentiment > 0.05 else "negative"

print("Top 5 Parts of Speech and Their Frequencies:")
for tag, freq in common_pos:
    print(f"{tag}: {freq}")

print("\nTop 30 Lemmatized Tokens:")
print(lemmatized_tokens)

print("\nAverage Sentiment Score:", avg_sentiment)
print("Overall Text Sentiment:", overall_sentiment)

plt.show()