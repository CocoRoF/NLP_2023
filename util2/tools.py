import pandas as pd
import math

def insert_text_after(target_string, search_text, insert_text):
    index = target_string.find(search_text)
    
    if index == -1:
        return target_string

    return target_string[:index + len(search_text)] + insert_text + target_string[index + len(search_text):]

def up_ten(n):
    return math.ceil(n / 10.0) * 10

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