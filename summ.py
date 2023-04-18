import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub
import tensorflow_text

# Load pre-trained word embedding model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

max_length = 0

def read_article(text, min_length=10, max_length=100, keywords=None):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stop words and filter sentences by length and keywords
    stop_words = stopwords.words('english')
    cleaned_sentences = []
    for sentence in sentences:
        words = sentence.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_sentence = ' '.join(filtered_words)
        sentence_length = len(cleaned_sentence.split())
        if sentence_length >= min_length and sentence_length <= max_length:
            if keywords is None:
                cleaned_sentences.append(cleaned_sentence)
            else:
                for keyword in keywords:
                    if keyword in sentence.lower():
                        cleaned_sentences.append(cleaned_sentence)
                        break
    
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


# def generate_summary(text, num_sentences=15, min_length=10, max_length=1000  , keywords=None):
#     # Clean the text and tokenize it into sentences
#     sentences = read_article(text, min_length=min_length, max_length=max_length, keywords=keywords)

#     # Filter out short and long sentences
#     sentences = [sent for sent in sentences if len(sent) >= min_length and len(sent) <= max_length]

#     # Build the similarity matrix and apply PageRank to get sentence scores
#     similarity_matrix = build_similarity_matrix(sentences)
#     sentence_scores = np.sum(similarity_matrix, axis=1)
#     sentence_scores = sentence_scores / np.max(sentence_scores) # Normalize scores
#     damping_factor = 0.85
#     for _ in range(10): # Run PageRank for 10 iterations
#         sentence_scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix, sentence_scores)

#     # Sort the sentences by their score and get the top N sentences
#     ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
#     top_sentences = ranked_sentences[:num_sentences]

#     # Include the top sentences that cover at least 25% of the text
#     summary_sentences = []
#     summary_length = 0
#     for score, sentence in top_sentences:
#         if summary_length < 0.5 * len(text.split()):
#             summary_sentences.append(sentence)
#             summary_length += len(sentence.split())

#     # Join the summary sentences into a single summary paragraph
#     summary = ' '.join(summary_sentences)

#     return summary


def generate_summary(text, num_sentences=15, min_length=10, max_length=100, max_summary_length=None, keywords=None):
    # Clean the text and tokenize it into sentences
    sentences = read_article(text, min_length=min_length, max_length=max_length, keywords=keywords)

    # Filter out short and long sentences
    sentences = [sent for sent in sentences if len(sent) >= min_length and len(sent) <= max_length]

    # Build the similarity matrix and apply PageRank to get sentence scores
    similarity_matrix = build_similarity_matrix(sentences)
    sentence_scores = np.sum(similarity_matrix, axis=1)
    sentence_scores = sentence_scores / np.max(sentence_scores) # Normalize scores
    damping_factor = 0.85
    for _ in range(10): # Run PageRank for 10 iterations
        sentence_scores = (1 - damping_factor) + damping_factor * np.dot(similarity_matrix, sentence_scores)

    # Sort the sentences by their score and get the top N sentences
    ranked_sentences = sorted(((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = ranked_sentences[:num_sentences]
    
    # Include the top sentences that cover at least 25% of the text
    summary_sentences = []
    summary_length = 0
    max_summary_length_words = max_summary_length or float('inf')
    for score, sentence in top_sentences:
        if summary_length < max_summary_length_words:
            summary_sentences.append(sentence)
            summary_length += len(sentence.split())
        else:
            break
    
    # Combine the summary sentences into a single string
    summary = ' '.join(summary_sentences)
    
    return summary


text = "The internet has become an essential part of our lives. With just a few clicks, we can access information, connect with friends and family, shop, and even work remotely. However, the internet is also a double-edged sword, and with all its advantages, it poses significant risks to our privacy and security.In recent years, cyberattacks have become more frequent and sophisticated, targeting both individuals and organizations. These attacks can result in financial loss, identity theft, and even reputational damage. That's why it's essential to take measures to protect yourself and your sensitive information.One of the most effective ways to protect yourself is by using strong, unique passwords for all your online accounts. A strong password is one that is long, complex, and unique to each account. Avoid using common words, phrases, or personal information that can be easily guessed. Instead, use a combination of uppercase and lowercase letters, numbers, and symbols.Another important measure is to enable two-factor authentication whenever possible. Two-factor authentication adds an extra layer of security to your accounts by requiring a second form of authentication, such as a text message or a biometric scan.You should also be cautious of phishing scams, which are attempts by hackers to steal your personal information by tricking you into clicking on a malicious link or downloading a harmful attachment. Always verify the sender's identity and be wary of unsolicited emails or messages.Finally, it's crucial to keep your software up-to-date. Software updates often include security patches that fix vulnerabilities and protect against known threats. So, make sure you regularly update your operating system, web browsers, and other software.In conclusion, the internet offers many benefits, but it also poses significant risks to our privacy and security. By taking the necessary precautions, such as using strong passwords, enabling two-factor authentication, being cautious of phishing scams, and keeping your software up-to-date, you can protect yourself and your sensitive information from cyber threats."
summary = generate_summary(text, num_sentences=15, min_length=10, max_length=1000, max_summary_length=500, keywords=None)
print(summary)
