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
          "emotion" : {"type" : "string", "enum" : ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']},
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
        {"review" : "고객의 감정 = Happiness, 고객의 리뷰 = 믿고쓰는 상품! 너무나도 만족합니다. 항상 좋은 제품 제공해주셔서 너무나도 감사합니다. 배송도 빠르고 서비스도 너무 좋아요 ^^"},
        {"output" : "안녕하세요 고객님, 고객님의 행복이 저희에게도 큰 행복입니다! 항상 저희 제품을 사용해주셔서 대단히 감사합니다. 앞으로도 좋은 서비스로 보답할 수 있도록 하겠습니다. 고맙습니다."})
    else:
      positive_memory = load_memory

    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
    
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("긍정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객에 대한 감사함을 표현하세요. 고객의 감정을 고려하여 답변하세요. 고객의 감정은 ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']가 존재합니다."),
          HumanMessagePromptTemplate.from_template("고객의 감정 = {user_emot}, 고객의 리뷰 = {review}")
        ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("긍정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객에 대한 감사함을 표현하세요. 고객의 감정을 고려하여 답변하세요. 고객의 감정은 ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']가 존재합니다."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("고객의 감정 = {user_emot}, 고객의 리뷰 = {review}")
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
          SystemMessagePromptTemplate.from_template("긍정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객에 대한 감사함을 표현하세요."),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("긍정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객에 대한 감사함을 표현하세요."),
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
        {"review" : "고객의 감정 = Contempt, 고객의 리뷰 = 예쁘고 심플해서 샀는데. 재질이 깔끄러워요. 살에 자국 다 베이고ㅠㅠ....폭망이에요. 재대로 확인안한 제 잘못이죠;;; 참고로 싱글세트 2. 퀸세트 1 샀습니다."},
        {"output" : "안녕하세요 고객님, 먼저 저희 제품을 선택해 주신 것에 대한 감사함을 먼저 표합니다. 싱글세트와 퀸세트를 구매해주셨는데, 의도치않게 불편을 드리게 되어 정말로 죄송합니다. 추후에는 이러한 부분을 보완하여 더욱 좋은 상품을 제공할 수 있도록 노력하겠습니다. 감사합니다."})

    else:
      negative_memory = load_memory
      
    llm = ChatOpenAI(model_name='gpt-4', temperature = temperature)
    
    if load_memory == None:
      prompt = ChatPromptTemplate(
        input_variables=["user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("부정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객의 마음을 이해하고, 고객에 대한 죄송함을 표현하세요. 고객의 감정을 고려하여 답변하세요. 고객의 감정은 ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']가 존재합니다."),
          HumanMessagePromptTemplate.from_template("고객의 감정 = {user_emot}, 고객의 리뷰 = {review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("부정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객의 마음을 이해하고, 고객에 대한 죄송함을 표현하세요. 고객의 감정을 고려하여 답변하세요. 고객의 감정은 ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise']가 존재합니다."),
          MessagesPlaceholder(variable_name="chat_history"),
          HumanMessagePromptTemplate.from_template("고객의 감정 = {user_emot}, 고객의 리뷰 = {review}")
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
          SystemMessagePromptTemplate.from_template("부정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객의 마음을 이해하고, 고객에 대한 죄송함을 표현하세요."),
          HumanMessagePromptTemplate.from_template("{review}")
          ]
      )
    else:
      prompt = ChatPromptTemplate(
        input_variables=["chat_history", "user_emot", "review"],
        messages = [
          SystemMessagePromptTemplate.from_template("부정적인 리뷰에 답하는 AI 봇으로서 주어지는 리뷰에 대한 답변을 작성하세요. 고객의 마음을 이해하고, 고객에 대한 죄송함을 표현하세요."),
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
    
def user_purpose(text:str, temperature:float=0) -> dict[str]:
  """
  의도분석을 수행하는 모듈. (1차 분류)
  몇 가지 논문을 참고하여 고객의 리뷰의도를 다음과 같이 구분함.
  1. 본인의 긍정적 경험에 대한 공유
  2. 공급자에 대한 칭찬
  3. 회사 및 제품에 대한 비난/비판 (2차 분류로 이어짐)
  4. 다른 고객에게 부정적 정보를 전달(concern for other consumers)
  """
  chat_LLM = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  schema = {
      "properties": {
          "purpose" : {"type" : "string", "enum" : ['For Sharing Positive Experience', 'For Helping The Company', 'For Venting Negative Feeling', 'For Concerning for Other Consumers'], "description" : 'Describes why consumers write the reviews'},
      },
      "required" : ["purpose"]
  }
  user_purpose_chain = create_tagging_chain(schema, chat_LLM)
  answer = user_purpose_chain.run(text)

  return answer

def specific_negative_purpose(text:str, temperature:float=0) -> dict[str, int]:
  """
  만약 비난/비판 관련 의도로 파악된 경우 세부적인 목적을 구분하기 위한 모듈. (2차 분류)
  1. 질나쁜 서비스에 대한 복수 및 가해. (나쁜 의도의 리뷰)
  2. 서비스 제공자에게 더욱 친절한 서비스를 이끌어 내기 위함. (Customers want more upfront policies)
  3. 다른 사람들과 힘을 모으기 위함. (나쁜 서비스가 한 번이 아니라, 여러번 이뤄지고 있음을 강력하게 지지하기 위함)
  4. 환불을 원하는 경우.
  5. 회사의 사과를 원하는 경우.
  6. 특정 문제에 대한 해결을 하기 위함. (3차 분류로 이어짐. 특정 제품/특정 서비스 한정)
  """
  chat_LLM = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  schema = {
      "properties": {
          "negative_purpose" : {"type" : "string", "enum" : ['For Harm and Vengeance', 'For Taking More Accommodating Service', 'For Collective Power from other Consumers', 'For taking Refund', 'For taking Apology', 'For Solving Specific Problem'], "description" : "Describes why consumers write Negative Purpose's reviews"},
      },
      "required" : ["negative_purpose"]
  }
  specific_negative_purpose_chain = create_tagging_chain(schema, chat_LLM)
  answer = specific_negative_purpose_chain.run(text)

  return answer

def problem_schema(problem_list:list) -> dict:
    problem = {'type' : 'string', 'enum' : problem_list, 'description' : "Classify which problem do consumers have"}
    problem_dict = {'problem' : problem}
    schema = {'properties' : problem_dict, 'required' : ['problem']}
    
    return schema

def specific_problem(text:str, temperature:float=0, problem_list:list = None) -> dict[str, int]:
  """
  특정 제품이나 서비스 한정으로 발생하는 문제들을 구분하기 위한 모듈. (3차 분류)
  해당 제품 서비스에서 빈번하게 발생하는 서비스에 대한 schema를 생성하여 Input으로 입력해야 함.
  """
  chat_LLM = ChatOpenAI(model_name='gpt-4', temperature=temperature)
  schema = problem_schema(problem_list)
  specific_negative_purpose_chain = create_tagging_chain(schema, chat_LLM)
  answer = specific_negative_purpose_chain.run(text)

  return answer

def total_purpose_classifier(text:str, temperature:float=0, problem_list:list = None) -> dict:
      purpose = None
      negative_purpose = None
      problem = None
      
      purpose = user_purpose(text, temperature = temperature)['purpose']
      if purpose == "For Venting Negative Feeling":
        negative_purpose = specific_negative_purpose(text, temperature = temperature)['negative_purpose']
        
        if negative_purpose == "For Solving Specific Problem":
          problem = specific_problem(text, temperature = temperature)['problem']
      
      answer = {'user_purpose' : purpose, 'negative_purpose' : negative_purpose, 'problem' : problem}
       
      return answer