from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders.csv_loader import CSVLoader


def sentiment_chat(text:str, temperature:float=0) -> dict[str, int]:
  """
  감성분석을 수행하는 모듈. 기본적인 Temperature를 0으로 설정해서, 대부분의 경우에서 동일한 답변을 수행하도록 하였음
  해당 모듈은 총 3가지로 감정을 분류하고, 각 감정의 세기를 표현하려 함.
  """
  senti_chat = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  schema = {
      "properties": {
          "sentiment" : {"type" : "string", "enum" : ['Positive', 'Neutral', 'Negative']},
          "aggressiveness": {"type" : 'integer', "enum" : [1,2,3,4,5], "description" : "describes how aggressive the statement is, the higher the more aggressive"}
      },
      "required" : ["sentiment", "aggressiveness"]
  }
  senti_chain = create_tagging_chain(schema, senti_chat)
  answer = senti_chain.run(text)

  return answer

def emotion_chat(text:str, temperature:float=0) -> dict[str, int]:
  """
  감정분석을 수행하는 모듈. 기본적인 Temperature를 0으로 설정해서, 대부분의 경우에서 동일한 답변을 수행하도록 하였음
  해당 모듈은 총 7가지로 감정을 분류하고, 각 감정의 세기를 표현하려 함.
  다중 정답이 필요한 경우가 존재할 수 있으니 이를 개선할 필요성이 존재.
  """
  emo_chat = ChatOpenAI(model_name='gpt-4', temperature = temperature)
  schema = {
      "properties" : {
          "emotion" : {"type" : "string", "enum" : ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', "Surprise"]},
          "aggressiveness": {"type" : 'integer', "enum" : [1,2,3,4,5], "description" : "describes how aggressive the statement is, the higher the more aggressive"}
      }
  }
  emo_chain = create_tagging_chain(schema, emo_chat)
  answer = emo_chain.run(text)

  return answer

def positive_chat(text:str, emotion:bool=False, temperature:float=0.8, emot_tempe:float=0, verbose:bool=False, load_memory='Default') -> str:
  """
  긍정적 채팅에 대한 리뷰 답변을 생성하는 모듈.
  기본적으로 감성 정보만을 받아서 구분되며, 이후 감정 정보를 고려하도록 설정할 수 있음.
  만약 감성 정보를 고려하지 않게 설정하면 감성 정보만 이용하여 수행함.
  memory는 기존 대화를 불러오는 Module을 의미함. 여기서는 default로 만들어진 대화를 참조하여 Few-shot Learning 되어있음.
  """
  if emotion:
    user_emot = emotion_chat(text, temperature = emot_tempe)['emotion']

    if load_memory == 'Default':
      positive_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, input_key="review")
      positive_memory.save_context(
        {"review" : "사용자의 감정 = Happiness, 사용자의 리뷰 = 믿고쓰는 상품! 너무나도 만족합니다. 항상 좋은 제품 제공해주셔서 너무나도 감사합니다. 배송도 빠르고 서비스도 너무 좋아요 ^^"},
        {"output" : "안녕하세요 고객님, 고객님의 행복이 저희에게도 큰 행복입니다! 항상 저희 제품을 사용해주셔서 대단히 감사합니다. 앞으로도 좋은 서비스로 보답할 수 있도록 하겠습니다. 고맙습니다."})
    else:
      positive_memory = load_memory

    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
    
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 긍정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 사용자의 칭찬에 감사와 고마움을 표하는 답변을 작성하세요. 사용자의 감정을 고려하여 답변하세요."),
          HumanMessagePromptTemplate.from_template("사용자의 감정 = {user_emot}, 사용자의 리뷰 = {review}")
        ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 긍정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 사용자의 칭찬에 감사와 고마움을 표하는 답변을 작성하세요. 사용자의 감정을 고려하여 답변하세요."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("사용자의 감정 = {user_emot}, 사용자의 리뷰 = {review}")
        ]
      )

    conversation = LLMChain(
      llm=llm,
      prompt=prompt,
      verbose=verbose,
      memory=positive_memory
    )

    answer = conversation({"user_emot" : user_emot, "review" : text})['text']
    return answer

  else:
    if load_memory == 'Default':
      positive_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, input_key="review")
      positive_memory.save_context(
        {"review" : "믿고쓰는 상품! 너무나도 만족합니다. 항상 좋은 제품 제공해주셔서 너무나도 감사합니다. 배송도 빠르고 서비스도 너무 좋아요 ^^"},
        {"output" : "안녕하세요 고객님, 항상 저희 제품을 사용해주셔서 대단히 감사합니다. 앞으로도 좋은 서비스로 보답할 수 있도록 하겠습니다. 고맙습니다."})
    else:
      positive_memory = load_memory
      
    
    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
  
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 긍정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 상대방의 칭찬에 감사와 고마움을 표하는 답변을 작성하세요."),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 긍정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 상대방의 칭찬에 감사와 고마움을 표하는 답변을 작성하세요."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )

    conversation = LLMChain(
      llm=llm,
      prompt=prompt,
      verbose=verbose,
      memory=positive_memory
    )

    answer = conversation({"review" : text})['text']
    return answer

def negative_chat(text:str, emotion:bool=False, temperature:float=0.8, emot_tempe:float=0, verbose:bool=False, load_memory='Default') -> str:
  """
  부정적 채팅에 대한 리뷰 답변을 생성하는 모듈.
  기본적으로 감성 정보만을 받아서 구분되며, 이후 감정 정보를 고려하도록 설정할 수 있음.
  만약 감성 정보를 고려하지 않게 설정하면 감성 정보만 이용하여 수행함.
  memory는 기존 대화를 불러오는 Module을 의미함. 여기서는 default로 만들어진 대화를 참조하여 Few-shot Learning 되어있음.
  """
  if emotion:
    user_emot = emotion_chat(text, temperature = emot_tempe)['emotion']

    if load_memory == 'Default':
      negative_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, input_key="review")
      negative_memory.save_context(
        {"review" : "사용자의 감정 = Contempt, 사용자의 리뷰 = 예쁘고 심플해서 샀는데. 재질이 깔끄러워요. 살에 자국 다 베이고ㅠㅠ....폭망이에요. 재대로 확인안한 제 잘못이죠;;; 참고로 싱글세트 2. 퀸세트 1 샀습니다."},
        {"output" : "안녕하세요 고객님, 먼저 저희 제품을 선택해 주신 것에 대한 감사함을 먼저 표합니다. 싱글세트와 퀸세트를 구매해주셨는데, 의도치않게 불편을 드리게 되어 정말로 죄송합니다. 추후에는 이러한 부분을 보완하여 더욱 좋은 상품을 제공할 수 있도록 노력하겠습니다. 감사합니다."})

    else:
      negative_memory = load_memory
      
    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
    
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 부정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 고객의 마음을 이해하고 위로하는 답변을 작성하세요. 사용자의 감정을 고려하여 답변하세요."),
          HumanMessagePromptTemplate.from_template("사용자의 감정 = {user_emot}, 사용자의 리뷰 = {review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 부정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 고객의 마음을 이해하고 위로하는 답변을 작성하세요. 사용자의 감정을 고려하여 답변하세요."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("사용자의 감정 = {user_emot}, 사용자의 리뷰 = {review}")
          ]
      )

    conversation = LLMChain(
      llm=llm,
      prompt=prompt,
      verbose=verbose,
      memory=negative_memory
    )

    answer = conversation({"user_emot" : user_emot, "review" : text})['text']
    return answer

  else:
    if load_memory == 'Default':
      negative_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True, input_key="review")
      negative_memory.save_context(
        {"review" : "예쁘고 심플해서 샀는데. 재질이 깔끄러워요. 살에 자국 다 베이고ㅠㅠ....폭망이에요. 재대로 확인안한 제 잘못이죠;;; 참고로 싱글세트 2. 퀸세트 1 샀습니다."},
        {"output" : "안녕하세요 고객님, 먼저 저희 제품을 선택해 주신 것에 대한 감사함을 먼저 표합니다. 싱글세트와 퀸세트를 구매해주셨는데, 의도치않게 불편을 드리게 되어 정말로 죄송합니다. 추후에는 이러한 부분을 보완하여 더욱 좋은 상품을 제공할 수 있도록 노력하겠습니다. 감사합니다."})
    else:
      negative_memory = load_memory
      
    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
    
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 부정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 고객의 마음을 이해하고 위로하는 답변을 작성하세요."),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("당신은 부정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 고객의 마음을 이해하고 위로하는 답변을 작성하세요."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )

    conversation = LLMChain(
      llm=llm,
      prompt=prompt,
      verbose=verbose,
      memory=negative_memory
    )

    answer = conversation({"review" : text})['text']
    return answer
  
def data_loader_csv(path:str, argument:dict):
    loader = CSVLoader(file_path = path, csv_args=argument)
    data = loader.load()  
