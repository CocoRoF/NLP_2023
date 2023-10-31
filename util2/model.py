import os
import pandas as pd
from util2.module import *
from util.module import *

class DataAnalyzer():
    def __init__(self, dataframe:pd.DataFrame):
        self.df = dataframe
        
    def Analysis(self, method:str = 'LCEL', save_path:str = None, analyzer_prompt_number:int = 0, analyzer_temperature:float = 0):
        with open('./Openai_API_Key.txt', 'r') as api:
            os.environ["OPENAI_API_KEY"] = api.read()
            
        for num in self.df.index:
            review = self.df.loc[num, 'user_review']
            
            if method == 'LCEL':
                result = review_analyzer(review, analyzer_prompt_number=analyzer_prompt_number, analyzer_temperature=analyzer_temperature)
                self.df.loc[num, 'user_sentiment'] = result['User_Sentiment']
                self.df.loc[num, 'user_emotion'] = result['User_Emotion']
                self.df.loc[num, 'user_intention'] = result['User_Intention']
            
            if method == 'NormLC':
                self.df.loc[num, 'user_sentiment'] = sentiment_chat(review, temperature=analyzer_temperature)['sentiment']
                self.df.loc[num, 'user_emotion'] = emotion_chat(review, temperature=analyzer_temperature)['emotion']
                self.df.loc[num, 'user_intention'] = total_purpose_classifier(review, temperature=analyzer_temperature)
            
            if save_path != None:
                self.df.to_excel(save_path, index=False)
                
        return self.df
            
    def Respond(self, method:str = 'LCEL', save_path:str = None, responder_prompt_number:int = 0, responder_temperature:float = 0, total_result:bool = False, col_name:str = 'AI_Response'):
        with open('./Openai_API_Key.txt', 'r') as api:
            os.environ["OPENAI_API_KEY"] = api.read()
        
        for num in self.df.index:
            review = self.df.loc[num, 'user_review']
            sentiment = self.df.loc[num, 'user_sentiment']
            emotion = self.df.loc[num, 'user_emotion']
            intention = self.df.loc[num, 'user_intention']
            
            if method == 'LCEL':
                result = responder(user_review=review, User_Sentiment=sentiment, User_Emotion=emotion, User_Intention=intention,responder_prompt_number=responder_prompt_number, responder_temperature=responder_temperature)
                self.df.loc[num, col_name] = result['Final_Response']
                
            if method == 'Norm':
                result = norm_responder(user_review=review, responder_temperature=responder_temperature)
                self.df.loc[num, col_name] = result
                
        return self.df