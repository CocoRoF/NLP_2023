import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def insert_text_after(target_string, search_text, insert_text):
    index = target_string.find(search_text)
    
    if index == -1:
        return target_string

    return target_string[:index + len(search_text)] + insert_text + target_string[index + len(search_text):]

def up_ten(n):
    return math.ceil(n / 10.0) * 10

def tfidf_cos_similarity(list_data:list):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(list_data)

    cosine_similarities = cosine_similarity(tfidf)
    avg_similarity = cosine_similarities.sum() / (cosine_similarities.shape[0] * cosine_similarities.shape[1] - cosine_similarities.shape[0])
    
    return avg_similarity

def embd_cos_similarity(list_data:list, model:SentenceTransformer):
    embeddings = model.encode(list_data, convert_to_tensor=True)
    cosine_similarities = cosine_similarity(embeddings)

    # 모든 유사도의 평균을 계산하기
    avg_similarity = cosine_similarities.sum() / (cosine_similarities.shape[0] * cosine_similarities.shape[1] - cosine_similarities.shape[0])
    return avg_similarity


                
def response_parser(s):
    lines = s.split('\n')
    result = {}
    
    result['Responding_Sentiment'] = lines[0].split('"')[1]
    result['Sentiment_reason'] = lines[0].split('(')[2].split(')')[0]
    
    result['Responding_Emotion'] = lines[1].split('"')[1]
    result['Emotion_reason'] = lines[1].split('(')[2].split(')')[0]
    
    result['Responding_Intention'] = lines[2].split('"')[1]
    result['Intention_reason'] = lines[2].split('(')[2].split(')')[0]
    
    result['Final_response'] = lines[4].split('= ')[1]
    
    return result