import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from langchain import OpenAI

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE

load_dotenv(find_dotenv())

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

# initialise session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []  # output
if "past" not in st.session_state:
    st.session_state["past"] = []  # past
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


def get_text():
    """
    Get the user input.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               placeholder="Your Data Science Assistant here, ask me anything...", label_visibility="hidden")
    return input_text

# --------------------------------------------------------------------------
# user input text field for encripted API key
# api = st.sidebar.input_text("Insert your API key here: ", type="password")

# expander for gpt models to choose from
# MODEL = st.sidebar.selectbox(label='Model',
#                              options=['gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002'])

# if api:
#     llm = OpenAI(
#         temperature=0,
#         openai_api_key=api,
#         model_name=MODEL
#     )
# --------------------------------------------------------------------------


# create conversation memory. And both conversation memory & conservation chain blocks are inside the above if statement, and there is else: st.write('No API found') after the 2 blocks
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationEntityMemory(
        llm=llm, k=10)

# create conversation chain
Conversation = ConversationChain(
    llm=llm,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=st.session_state.conversation_memory,
)

input_text = get_text()

if input_text:
    output = Conversation.run(input=input_text)

    st.session_state.past.append(input_text)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i], icon="🎁")
