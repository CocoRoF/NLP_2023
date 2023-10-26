from langchain.prompts import ChatPromptTemplate

def response_prompt_selector(prompt_num:int):
    if prompt_num == 0:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """As a marketing manager managing online customer reviews, write a response to the following 'Review'.\n\nWhen composing your reply, it is important to keep 'Customer Sentiment' in mind.\nWhen composing your reply, it is important to keep 'Customer Emotion' in mind.\nWhen composing your reply, it is important to keep 'Customer Intention' in mind.\n\nYour answer must follow this 'Format' below.\nFormat:\n(Responding to Customer Sentiment) + "(The sentence in the customer review that is the reason for generating that response.)"\n(Responding to Customer Emotion) + "(The sentence in the customer review that is the reason for generating that response.)"\n(Responding to Customer Intention) + "(The sentence in the customer review that is the reason for generating that response.)"\n\nThe Final Generated Response to that Customer Review = Your Final Response"""),
                ("human", "Customer Sentiment:\n{customer_sentiment}\n\nCustomer Emotion:\n{customer_emotion}\n\nCustomer Intention:\n{customer_intention}\n\nReview:\n{review}")
            ]
        )
    
    return prompt


def analysis_prompt_selector(prompt_num:int):
    if prompt_num == 0:
        function_prompt = [
            {
                "name": "Describer",
                "description": "Analyze the following 'Review'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "User_Sentiment": {
                            "type": "string",
                            "enum": ['Positive', 'Neutral', 'Negative'],
                            "description": "Sentiment Analysis of the 'Review'"
                        },
                        "User_Emotion": {
                            "type": "string",
                            "enum": ['Anger', 'Disgust', 'Fear', 'Happiness', 'Contempt', 'Sadness', 'Surprise'],
                            "description": "Emotion Analysis of the 'Review'"
                        },
                        "User_Intention": {
                            "type": "string",
                            "description": "Intention Analysis of the 'Review'"
                        },
                    },
                    "required": ["User_Sentiment", "User_Emotion", "User_Intention"]
                }
            }
        ]
    
    return function_prompt


def Response_output_selector(prompt_num:int):
    if prompt_num == 0:
        function_prompt = [
            {
                "name": "Responder",
                "description": "Respond appropriately to the following customer 'Reviews'",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Responding_to_Customer_Sentiment": {
                            "type": "string",
                            "description": "Responding to Customer Sentiment"
                        },
                        "Sentiment_Reason": {
                            "type": "string",
                            "description": "The sentence in the customer review that is the reason for generating that response to customer 'Sentiment'"
                        },
                        "Responding_to_Customer_Emotion": {
                            "type": "string",
                            "description": "Responding to Customer Emotion"
                        },
                        "Emotion_Reason": {
                            "type": "string",
                            "description": "The sentence in the customer review that is the reason for generating that response to customer 'Emotion'"
                        },
                        "Responding_to_Customer_Intention": {
                            "type": "string",
                            "description": "Emotion Analysis of the 'Review'"
                        },
                        "Intention_Reason": {
                            "type": "string",
                            "description": "The sentence in the customer review that is the reason for generating that response to customer 'Intention'"
                        },
                        "Final_Response": {
                            "type": "string",
                            "description": "The Final Generated Response to Customer Review"
                        },
                    },
                    "required": ["Responding_to_Customer_Sentiment", "Sentiment_Reason", "Responding_to_Customer_Emotion", "Emotion_Reason", "Responding_to_Customer_Intention", "Intention_Reason", "Final_Response"]
                }
            }
        ]
    
    return function_prompt
        