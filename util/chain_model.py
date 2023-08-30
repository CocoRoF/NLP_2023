from util.module import *

class Review_Chain_model():
      def __init__(self, text: str, Show_Variable:bool = False, verbose:bool = False, 
                   module_temperature:float = 0, answer_temperature:float = 0, problem_list:list = None, memory = 'Default'):
            self.review_text = text
            self.show_variable = Show_Variable
            self.verbose = verbose
            self.module_temperature = module_temperature
            self.answer_temperature = answer_temperature
            self.problem_list = problem_list
            self.memory = memory
            
            self.review_sentiment = sentiment_chat(self.review_text, temperature=self.module_temperature)['sentiment']
            self.review_emotion = emotion_chat(self.review_text, temperature=self.module_temperature)['emotion']
            self.user_purpose = total_purpose_classifier(self.review_text, temperature=self.module_temperature, problem_list=self.problem_list)
            self.review_purpose_1 = self.user_purpose['user_purpose']
            self.review_purpose_2 = self.user_purpose['negative_purpose']
            if self.review_purpose_2 == None:
                  self.review_purpose_2 = '없음'
            self.review_purpose_3 = self.user_purpose['problem']
            if self.review_purpose_3 == None:
                  self.review_purpose_3 = '없음'
            
            self.variables = {'Review' : self.review_text, 'Sentiment' : self.review_sentiment, 'Emotion' : self.review_emotion,
                              'Purpose' : self.review_purpose_1, 'Negative_Purpose' : self.review_purpose_2, 'Problem' : self.review_purpose_3}
      
      def show_variables(self):
            print(self.variables)
            
      def Run_Answer(self):
            if self.memory == 'Default':
                  self.memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, input_key="review")
                  self.memory.save_context(
                        {"review" : "고객의 감성 = Positive, 고객의 감정 = Happiness, 고객의 리뷰의도 = For Sharing Positive Experience, 부정리뷰의 이유 = 없음, 고객의 문제 사항 = 없음, 고객의 리뷰 = 믿고쓰는 상품! 너무나도 만족합니다. 항상 좋은 제품 제공해주셔서 너무나도 감사합니다. 배송도 빠르고 서비스도 너무 좋아요 ^^"},
                        {"output" : "안녕하세요 고객님, 고객님의 행복이 저희에게도 큰 행복입니다! 항상 저희 제품을 사용해주셔서 대단히 감사합니다. 앞으로도 좋은 서비스로 보답할 수 있도록 하겠습니다. 고맙습니다."})
            
            
            llm = ChatOpenAI(model_name='gpt-4', temperature = self.answer_temperature)

            prompt = ChatPromptTemplate(
                input_variables=['chat_history', 'user_sentiment', 'user_emotion', 'user_purpose', 'user_negative_review_reason', 'service_problem', 'review'],
                messages = [
                  SystemMessagePromptTemplate.from_template("고객의 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객의 감성을 고려하세요. 고객의 감성은 ['Positive', 'Negative']가 존재합니다. 고객의 감정을 고려하여 답변하세요. 고객의 감정은 ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']가 존재합니다. 고객의 리뷰의도를 고려하여 답변하세요. 고객의 리뷰의도는 ['For Sharing Positive Experience', 'For Helping The Company', 'For Venting Negative Feeling', 'For Concerning for Other Consumers']가 존재합니다. 고객의 리뷰의도가 'For Venting Negative Feeling'인 경우, 고객의 부정 리뷰의 이유를 고려하세요. 고객의 부정리뷰 의유는 ['For Harm and Vengeance', 'For Taking More Accommodating Service', 'For Collective Power from other Consumers', 'For taking Refund', 'For taking Apology', 'For Solving Specific Problem']가 존재합니다. 고객의 부정 리뷰 이유가 'For Solving Specific Problem'인 경우 문제 사항이 무엇인지 고려하세요. 문제 사항은 {}가 존재합니다." .format(self.problem_list)),
                  MessagesPlaceholder(variable_name="chat_history"),
                  HumanMessagePromptTemplate.from_template("고객의 감성 = {user_sentiment}, 고객의 감정 = {user_emotion}, 고객의 리뷰의도 = {user_purpose}, 부정리뷰의 이유 = {user_negative_review_reason}, 고객의 문제 사항 = {service_problem}, 고객의 리뷰 = {review}")
                ]
              )
              
            conversation = LLMChain(
                llm=llm,
                prompt=prompt,
                verbose=self.verbose,
                memory=self.memory
              )          
            
            answer = conversation({'user_sentiment' : self.review_sentiment, 'user_emotion' : self.review_emotion, 'user_purpose' : self.review_purpose_1, 'user_negative_review_reason' : self.review_purpose_2, 'service_problem' : self.review_purpose_3,  "review" : self.review_text})['text']
            return answer


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