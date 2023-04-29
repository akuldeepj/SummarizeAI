

#BASIC MODEL WORSE PERFORMANCE

# import nltk
# from nltk.tokenize import sent_tokenize
# from nltk.corpus import stopwords
# from nltk.cluster.util import cosine_distance
# import numpy as np
# import networkx as nx

# def read_article(text):
#     # Tokenize the text into sentences
#     sentences = sent_tokenize(text)
    
#     # Remove stop words and return a list of cleaned sentences
#     stop_words = stopwords.words('english')
#     cleaned_sentences = []
#     for sentence in sentences:
#         words = sentence.split()
#         filtered_words = [word for word in words if word.lower() not in stop_words]
#         cleaned_sentences.append(' '.join(filtered_words))
    
#     return cleaned_sentences

# def sentence_similarity(sent1, sent2):
#     # Convert the sentences to vectors using word embeddings
#     # You can use other methods to convert sentences to vectors as well
#     sent1 = nltk.word_tokenize(sent1)
#     sent2 = nltk.word_tokenize(sent2)
#     words = list(set(sent1 + sent2))
#     vec1 = [int(word in sent1) for word in words]
#     vec2 = [int(word in sent2) for word in words]
    
#     # Calculate the cosine similarity between the vectors
#     return 1 - cosine_distance(vec1, vec2)

# def build_similarity_matrix(sentences):
#     # Create an empty similarity matrix
#     n = len(sentences)
#     similarity_matrix = np.zeros((n, n))
    
#     # Fill the similarity matrix with pairwise sentence similarities
#     for i in range(n):
#         for j in range(i+1, n):
#             similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
#     return similarity_matrix

# def generate_summary(text, num_sentences=9):
#     # Clean the text and tokenize it into sentences
#     sentences = read_article(text)
    
#     # Build the similarity matrix and apply PageRank to get sentence scores
#     similarity_matrix = build_similarity_matrix(sentences)
#     sentence_scores = nx.pagerank(nx.from_numpy_array(similarity_matrix))
    
#     # Sort the sentences by their score and get the top N sentences
#     ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
#     summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]
    
#     # Combine the top N sentences into a summary
#     summary = ' '.join(summary_sentences)
    
#     return summary



# DECENT PERFORMANCE

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub
import tensorflow_text
from grammar import grammify


# Load pre-trained word embedding model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

def read_article(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stop words and return a list of cleaned sentences
    stop_words = stopwords.words('english')
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentences.append(' '.join(filtered_words))
    
    return cleaned_sentences

def sentence_similarity(sent1, sent2):
    # Convert the sentences to vectors using pre-trained word embeddings
    vec1 = embed([sent1])[0].numpy()
    vec2 = embed([sent2])[0].numpy()
    
    # Calculate the cosine similarity between the vectors
    return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]

def build_similarity_matrix(sentences):
    # Create an empty similarity matrix
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    # Fill the similarity matrix with pairwise sentence similarities
    for i in range(n):
        for j in range(i+1, n):
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
            similarity_matrix[j][i] = similarity_matrix[i][j]
    
    return similarity_matrix

def generate_summary(text, num_sentences=9):
    # Clean the text and tokenize it into sentences
    sentences = read_article(text)
    
    # Build the similarity matrix and apply PageRank to get sentence scores
    similarity_matrix = build_similarity_matrix(sentences)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    sentence_scores = sentence_scores / np.max(sentence_scores) # Normalize scores
    damping_factor = 0.85
    for _ in range(10): # Run PageRank for 10 iterations
        sentence_scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix, sentence_scores)
    
    # Sort the sentences by their score and get the top N sentences
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    summary_sentences = [s for score, s in ranked_sentences[:num_sentences]]
    
    # Combine the top N sentences into a summary
    summary = ' '.join(summary_sentences)
    
    return summary



