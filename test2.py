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

    You are a helpful IT support agent, helping clients by following directives and answering questions related to technical issues they are having.

    Generate your response by following the steps below:

    1. Recursively break-down the question into smaller questions/directives/IT related issues they are having

    2. For each atomic question/directive:

    2a. Select the most relevant information from the context on the IT related issue in light of the conversation history

    3. Generate a draft response using the selected information, including all possible Fix action taken (detailed) in the context

    4. Remove duplicate content from the draft response but ensure to keep all possible Fix action taken (detailed)

    5. Generate your final response after adjusting it to increase accuracy and relevance

    6. Now only show your final response! Do not provide any explanations or details
    
    7. If a customer does not ask a question or provide any directive, politely respond

    CONTEXT:

    """
        + str(context)
        + """
    """
    )

    user = (
        """
    This is the context: """
        + str(context)
        + """
    End of context.

    Based on the context, answer the following question: """
        + question
        + """
    Answer: """
    )

    messages = [
        ChatMessage(content=system, role="system"),
        ChatMessage(content=user, role="user"),
    ]

    return messages


############ Jamba RAG + Chat


def jamba_rag(question):
    chat_hist.add_user_message(question)
    context = client.library.search.create(query=question, labels=["Orange Doc 3"])
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

##Visuals

logo_url = "data:image/png;base64,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"

st.image(logo_url, width=300)

st.title("OrangeChat - AI21 Business Assistant")

st.write("This application will allow you to chat with your documents")

##App run logic


st.header("Agent Chat and Troubleshoot with Your Knowledge Base")
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
