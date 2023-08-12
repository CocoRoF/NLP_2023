from util.module import *

def sentiment_chain_model(text:str, show_sentiment:bool=False, emotion:bool=False, show_emotion:bool=False, senti_tempe:float=0, emot_tempe:float=0, chat_tempe:float=0.8, verbose:bool=False, load_memory='Default'):
  user_chat = text
  sentiment = sentiment_chat(user_chat, temperature=senti_tempe)['sentiment']
  if emotion:
    user_emotion = emotion_chat(user_chat, temperature=emot_tempe)['emotion']
    if sentiment == 'Positive':
      answer = positive_chat(user_chat, emotion = emotion, temperature=chat_tempe, emot_tempe=emot_tempe, verbose=verbose, load_memory=load_memory)

    else:
      answer = negative_chat(user_chat, emotion = emotion, temperature=chat_tempe, emot_tempe=emot_tempe, verbose=verbose, load_memory=load_memory)

    result_dict = {}
    if show_sentiment:
        result_dict['Sentiment'] = sentiment
    if show_emotion:
        result_dict['Emotion'] = user_emotion
    result_dict['Answer'] = answer

    return result_dict

  else:
    if sentiment == 'Positive':
      answer = positive_chat(user_chat, temperature=chat_tempe, emot_tempe=emot_tempe, verbose=verbose, load_memory=load_memory)

    else:
      answer = negative_chat(user_chat, temperature=chat_tempe, emot_tempe=emot_tempe, verbose=verbose, load_memory=load_memory)

    result_dict = {}
    if show_sentiment:
        result_dict['Sentiment'] = sentiment
    result_dict['Answer'] = answer

    return result_dict