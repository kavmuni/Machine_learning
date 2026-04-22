from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Sample documents
documents = ["Data Science is fun", "Data Science is about data", "Enjoy the fun"]

# Initialize and transform
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Print word features
print(vectorizer.get_feature_names_out())
# Print TF-IDF matrix
print(tfidf_matrix.toarray())


df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
print(df) # prints single row as DataFrame
