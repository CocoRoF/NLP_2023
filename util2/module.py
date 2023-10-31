from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from util2.prompt import analysis_prompt_selector, response_prompt_selector, Response_output_selector
import os

# API Load
with open('./Openai_API_Key.txt', 'r') as api:
    os.environ["OPENAI_API_KEY"] = api.read()
    
def review_analyzer(user_review:str, analyzer_prompt_number:int = 0, analyzer_temperature:float = 0):
    function_prompt = analysis_prompt_selector(prompt_num=analyzer_prompt_number)
    analyzer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "As an online hotel manager, analyze customer reviews."),
            ("human", "{analysis}")
        ]
    )
    
    analyzer_model = ChatOpenAI(model="gpt-4", temperature=analyzer_temperature).bind(function_call={"name": "Describer"}, functions=function_prompt)
    runnable = (
        {"analysis": RunnablePassthrough()} 
        | analyzer_prompt 
        | analyzer_model
        | JsonOutputFunctionsParser()
    )
    result = runnable.invoke(user_review)
    return result
    
    
def responder(user_review:str, responder_prompt_number:int = 0, responder_temperature:float = 0, User_Sentiment:str = None, User_Emotion:str = None, User_Intention:str = None):
    responder_prompt = response_prompt_selector(responder_prompt_number)
    responder_model = ChatOpenAI(model="gpt-4", temperature=responder_temperature)
    responder_function = Response_output_selector(prompt_num=0)
    
    response_chain = responder_prompt | responder_model.bind(function_call={"name" : "Responder"}, functions=responder_function) | JsonOutputFunctionsParser()
    response = response_chain.invoke(
        {"customer_sentiment" : User_Sentiment, "customer_emotion" : User_Emotion, "customer_intention" : User_Intention, "review" : user_review})
    
    return response

def norm_responder(user_review:str, responder_temperature:float = 0):
    responder_model = ChatOpenAI(model="gpt-4", temperature=responder_temperature)
    norm_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Respond to review"),
            ("human", "{review}")
        ]
    )
    responder_chain = norm_prompt | responder_model
    response = responder_chain.invoke({"review" : user_review})
    
    return response.content
    

