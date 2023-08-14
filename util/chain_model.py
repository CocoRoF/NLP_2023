from util.module import *

def sentiment_chain_model(text:str, show_sentiment:bool=False, emotion:bool=False, show_emotion:bool=False, senti_tempe:float=0, emot_tempe:float=0, chat_tempe:float=0.8, verbose:bool=False, load_memory='Default') -> dict:
  """
  감성/감정 정보를 바탕으로 답변을 생성하는 모듈
  emotion 인자가 True이면 감정 정보를 고려하도록 함.
  show_sentiemnt, show_emotion 인자에 따라 감성/감정 정보를 결과로 출력함
  memory의 인자는 langchain의 BaseMemory를 받아 고려하게 됨. 만약 BaseMemory가 None이면 Few-shot 없이 Zero-show으로 system prompt만 사용함.
  """
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