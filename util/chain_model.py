from util.module import *

def sentiment_chain_model(text:str, show_sentiment:bool=False, show_emotion:bool=False):
  user_chat = text
  sentiment = sentiment_chat(user_chat)['sentiment']
  emotion = emotion_chat(user_chat)['emotion']

  if sentiment == ('Very Positive') or ('Positive'):
    answer = positive_chat(user_chat)

  else:
    answer = negative_chat(user_chat)

  result_dict = {}
  if show_sentiment:
      result_dict['Sentiment'] = sentiment
  if show_emotion:
      result_dict['Emotion'] = emotion
  result_dict['Answer'] = answer

  return result_dict

def sentiment_chain_model(text:str, show_sentiment:bool=False, show_emotion:bool=False):
  user_chat = text
  sentiment = sentiment_chat(user_chat)['sentiment']
  emotion = emotion_chat(user_chat)['emotion']

  if sentiment == ('Very Positive') or ('Positive'):
    answer = positive_chat(user_chat)

  else:
    answer = negative_chat(user_chat)

  result_dict = {}
  if show_sentiment:
      result_dict['Sentiment'] = sentiment
  if show_emotion:
      result_dict['Emotion'] = emotion
  result_dict['Answer'] = answer

  return result_dict