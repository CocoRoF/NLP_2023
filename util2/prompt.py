from langchain.prompts import ChatPromptTemplate


def prompt_selector(prompt_num:int):
    if prompt_num == 0:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """As a marketing manager managing online customer reviews, write a response to the following 'Review'.\n\nWhen composing your reply, it is important to keep 'Customer Sentiment' in mind.\nWhen composing your reply, it is important to keep 'Customer Emotion' in mind.\nWhen composing your reply, it is important to keep 'Customer Intention' in mind.\n\nYour answer must follow this 'Format' below.\nFormat:\n(Responding to Customer Sentiment) + "(The sentence in the customer review that is the reason for generating that response.)"\n(Responding to Customer Emotion) + "(The sentence in the customer review that is the reason for generating that response.)"\n(Responding to Customer Intention) + "(The sentence in the customer review that is the reason for generating that response.)"\n\nThe Final Generated Response to that Customer Review = (Your Final Response)""")
                ("human", "Customer Sentiment:\n{customer_sentiment}\n\nCustomer Emotion:\n{customer_emotion}\n\nCustomer Intention:\n{customer_intention}\n\nReview:\n{review}")
            ]
        )
    
    return prompt
        