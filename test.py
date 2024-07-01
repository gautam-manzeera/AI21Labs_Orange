############# Import packages

import ai21
import langchain
import langchain_core
from ai21 import AI21Client
from ai21.models import ChatMessage, DocumentType, Penalty, RoleType, SummaryMethod
from ai21.models.chat import ChatMessage
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

############## Set API Key


client = AI21Client(api_key="J2DmhjI6ClpGcCw0qduuY3kkK9B8F4YP")

############# Instantiate Chat history

chat_hist = ChatMessageHistory()

############# Create Jamba Prompt


def create_messages(question, context):

    system = (
        """

    You are a helpful document support agent, helping team members by following directives and answering questions.

    Generate your response by following the steps below:

    1. Recursively break-down the question into smaller questions/directives

    2. For each atomic question/directive:

    2a. Select the most relevant information from the context in light of the conversation history

    3. Generate a draft response using the selected information

    4. Remove duplicate content from the draft response

    5. Generate your final response after adjusting it to increase accuracy and relevance

    6. Now only show your final response! Do not provide any explanations or details

    CONTEXT:

    """
        + str(context)
        + """
    """
    )

    messages = [
        ChatMessage(content=system, role="system"),
        ChatMessage(content=question, role="user"),
    ]

    return messages


############ Jamba RAG + Chat


def jamba_rag(question):
    chat_hist.add_user_message(question)
    context = client.library.search.create(query=question, labels=["Orange Doc 1"])
    messages = create_messages(chat_hist.messages[0].content, context)
    jamba_response = client.chat.completions.create(
        model="jamba-instruct", messages=messages, temperature=0.4
    )
    chat_hist.add_ai_message(jamba_response.choices[0].message.content)
    return jamba_response.choices[0].message.content


from io import StringIO

#########Streamlit App
import streamlit as st

# Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("OrangeChat - AI21 Business Assistant")

st.write("This application will allow you to chat with your documents")

##App run logic


st.header("Chat with Your Customer Operations Guide (COG) Documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if question := st.chat_input("What Question Do You Have?"):
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
        ans = jamba_rag(question)
    with st.chat_message("assistant"):
        st.markdown(ans)
    st.session_state.messages.append({"role": "assistant", "content": ans})
