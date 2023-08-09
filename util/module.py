from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain

def sentiment_chat(text:str):
  senti_chat = ChatOpenAI(model_name='gpt-4', temperature = 0)
  schema = {
      "properties": {
          "sentiment" : {"type" : "string", "enum" : ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Verey Negative']},
          "aggressiveness": {"type" : 'integer', "enum" : [1,2,3,4,5], "description" : "describes how aggressive the statement is, the higher the more aggressive"}
      },
      "required" : ["sentiment", "aggressiveness"]
  }
  senti_chain = create_tagging_chain(schema, senti_chat)
  answer = senti_chain.run(text)

  return answer

def emotion_chat(text:str):
  emo_chat = ChatOpenAI(model_name='gpt-4', temperature = 0)
  schema = {
      "properties" : {
          "emotion" : {"type" : "string", "enum" : ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', "Surprise"]},
          "aggressiveness": {"type" : 'integer', "enum" : [1,2,3,4,5], "description" : "describes how aggressive the statement is, the higher the more aggressive"}
      }
  }
  emo_chain = create_tagging_chain(schema, emo_chat)
  answer = emo_chain.run(text)

  return answer

def positive_chat(text:str):
  positive_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
  positive_memory.save_context({"input" : "믿고쓰는 상품! 너무나도 만족합니다. 항상 좋은 제품 제공해주셔서 너무나도 감사합니다. 배송도 빠르고 서비스도 너무 좋아요 ^^"},
   {"output" : "안녕하세요 고객님, 항상 저희 제품을 사용해주셔서 대단히 감사합니다. 앞으로도 좋은 서비스로 보답할 수 있도록 하겠습니다. 고맙습니다."})

  llm = ChatOpenAI(model_name='gpt-4', temperature = 0.8)
  prompt = ChatPromptTemplate(messages = [
      SystemMessagePromptTemplate.from_template("당신은 긍정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 상대방의 칭찬에 감사와 고마움을 표하는 답변을 작성하세요."),
      MessagesPlaceholder(variable_name="chat_history"),
      HumanMessagePromptTemplate.from_template("{question}")
      ]
  )

  conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=positive_memory
  )

  answer = conversation({"question" : text})['text']
  return answer

def negative_chat(text:str):
  negative_memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
  negative_memory.save_context({"input" : "예쁘고 심플해서 샀는데. 재질이 깔끄러워요. 살에 자국 다 베이고ㅠㅠ....폭망이에요. 재대로 확인안한 제 잘못이죠;;; 참고로 싱글세트 2. 퀸세트 1 샀습니다."},
   {"output" : "안녕하세요 고객님, 먼저 저희 제품을 선택해 주신 것에 대한 감사함을 먼저 표합니다. 싱글세트와 퀸세트를 구매해주셨는데, 의도치않게 불편을 드리게 되어 정말로 죄송합니다. 추후에는 이러한 부분을 보완하여 더욱 좋은 상품을 제공할 수 있도록 노력하겠습니다. 감사합니다."})

  llm = ChatOpenAI(model_name='gpt-4', temperature = 0.8)
  prompt = ChatPromptTemplate(messages = [
      SystemMessagePromptTemplate.from_template("당신은 부정적인 리뷰에 답변을 달아주는 유용한 AI 봇입니다. 고객의 마음을 이해하고 위로하는 답변을 작성하세요."),
      MessagesPlaceholder(variable_name="chat_history"),
      HumanMessagePromptTemplate.from_template("{question}")
      ]
  )

  conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False,
    memory=negative_memory
  )

  answer = conversation({"question" : text})['text']
  return answer
