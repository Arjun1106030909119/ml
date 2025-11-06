# ex05b.py - spam classification script recreated from the screenshot
#csv file
#Category,Message
#ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet..."
#ham,"Ok lar... Joking wif u oni..."
#spam,"Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply"
#ham,"U dun say so early hor... U c already then say..."
#ham,"Nah I don't think he goes to usf, he lives around here though"
#spam,"WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only"
#ham,"Even my brother is not like to speak with me. They treat me like aids patent."
#spam,"Urgent! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to 81010. T&C's apply"
#ham,"I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promises. You have been wonderful and a blessing at all times."
#spam,"FreeMsg: Txt: CALL to No: 86888 to claim your reward of 3 hours talk time to use from your phone now!"
#ham,"I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."
#spam,"Congrats! 1-year subscription to ringtone service. To unsubscribe text STOP to 87654. For info visit www.example.com"
#ham,"SIX chances to win CASH! From 100 to 20,000 pounds. Send STOP to 80062 to opt out."
#spam,"You have 1 new voicemail. Call 1234 to listen."
#ham,"I'll call you later."




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# 1) Load the data (assumes spam.csv is in the working directory)
# If your CSV has a different encoding or separators, adjust arguments accordingly.
df = pd.read_csv("spam.csv")
print(df.head())

# 2) Quick groupby description (optional)
print(df.groupby('Category')['Message'].describe())

# 3) Create binary target column 'spam' (1 if Category == 'spam', else 0)
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(df.head())

# 4) Train / test split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam, test_size=0.2, random_state=42)

# 5) CountVectorizer -> transform text to token counts
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train.values)
X_test_count = vectorizer.transform(X_test.values)

# 6) Train MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)

# Evaluate
score = model.score(X_test_count, y_test)
print("MultinomialNB accuracy (manual vectorizer):", score)

# 7) Pipeline version (vectorizer + classifier)
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)
pipeline_score = clf.score(X_test, y_test)
print("Pipeline accuracy:", pipeline_score)

# 8) Predict on new examples
emails = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
    "Hey John, are we still meeting for coffee tomorrow?"
]
preds = clf.predict(emails)
print("Predictions for example emails (0 = ham, 1 = spam):", preds)