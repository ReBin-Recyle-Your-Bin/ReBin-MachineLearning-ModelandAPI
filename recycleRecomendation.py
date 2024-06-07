import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dataset_recycling = None

def load_dataset():
  dataset_recycling = pd.read_excel('./dataset_recommendation_indo.xlsx')
  return dataset_recycling


def preprocessing_data(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d|\W|_', ' ', text)  # Correcting the regex pattern
    return text

def get_recommendation(keyword):
    global dataset_recycling
    
    if dataset_recycling is None:
        dataset_recycling = load_dataset()

    vectorizer = TfidfVectorizer(preprocessor=preprocessing_data)
    recycling_matrix = vectorizer.fit_transform(dataset_recycling['ingredients'])

    recycling_matrix.astype(np.float32)

    # Convert keyword to TF-IDF vector
    keyword_vector = vectorizer.transform([keyword])

    # Calculate cosine similarity between keyword vector and recycling matrix
    similarity_scores = cosine_similarity(keyword_vector, recycling_matrix)

    # Print cosine similarity scores
    print("Cosine Similarity Scores:")
    print(similarity_scores)
    print()

    top_recycling_indices = similarity_scores.argsort()[0][::-1]
    top_3_recycling = dataset_recycling.iloc[top_recycling_indices][:3]

    # Remove the index column before converting to dictionary
    top_3_recycling = top_3_recycling.reset_index(drop=True)  # Reset index without adding new index column

    recycling_cleaned = []
    for idx, row in top_3_recycling.iterrows():
        similarity_score = similarity_scores[0][idx]
        row_dict = row.to_dict()
        row_dict['cosine_similarity'] = similarity_score
        recycling_cleaned.append(row_dict)

    return recycling_cleaned
