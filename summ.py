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


text = 'Exercise is an important part of a healthy lifestyle, and it provides numerous benefits that can improve your physical, mental, and emotional wellbeing. Whether you are looking to improve your fitness level, lose weight, or simply feel better, regular exercise can help you achieve your goals.One of the most obvious benefits of regular exercise is improved physical health. Exercise helps to strengthen your muscles and bones, increase your cardiovascular health, and improve your overall physical endurance. This can help you to maintain a healthy weight, reduce your risk of chronic diseases such as heart disease, diabetes, and obesity, and improve your overall quality of life.In addition to the physical benefits, exercise also has numerous mental health benefits. Exercise has been shown to reduce symptoms of depression and anxiety, improve self-esteem and confidence, and reduce stress levels. This is because exercise triggers the release of endorphins, which are chemicals in the brain that produce feelings of happiness and pleasure. Regular exercise can help you to maintain a positive mood and improve your overall mental wellbeing.Another benefit of regular exercise is improved cognitive function. Exercise has been shown to improve brain function, including memory, attention, and problem-solving skills. This is because exercise increases blood flow to the brain, which can help to improve neural connections and promote the growth of new brain cells. Regular exercise can help you to maintain a sharp mind and improve your overall cognitive health.Exercise can also help to improve your sleep quality. Regular exercise can help you to fall asleep faster and stay asleep longer, which can help you to feel more rested and refreshed in the morning. Exercise has also been shown to reduce symptoms of sleep disorders such as insomnia, sleep apnea, and restless leg syndrome.Finally, exercise can be a great way to improve your social life and connect with others. Joining a fitness class or group can provide you with a supportive community and a sense of belonging. Exercise can also be a fun way to spend time with friends and family, and it can help you to meet new people and make new friends.Overall, regular exercise provides numerous benefits that can improve your physical, mental, and emotional wellbeing. Whether you are looking to improve your fitness level, lose weight, or simply feel better, incorporating regular exercise into your daily routine can help you achieve your goals and live a healthier, happier life. So, it\'s important to find a type of exercise that you enjoy and make it a regular part of your routine.'
summary = generate_summary(text, num_sentences=15, min_length=10, max_length=1000, max_summary_length=500, keywords=None)
print(summary)
