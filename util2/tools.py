import pandas as pd

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