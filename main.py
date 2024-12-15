from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferMemory

import os
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

load_dotenv(find_dotenv())
# apikey = os.getenv("OPENAI_API_KEY")

with open("styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.write("<br><br><br>", unsafe_allow_html=True)
    # st.write("<p style = 'text-align:center'>✨  Your weapon of choice  ✨</p>",
    #          unsafe_allow_html=True, divider='rainbow')
    st.subheader("Your weapon of choice", divider="rainbow")
    choice = st.radio(
        label="Choose your preferred LLM:", label_visibility="hidden",
        options=("Choose your preferred LLM:",
                 "gpt-3.5-turbo", "gpt-4", "gpt-4o"),
    )
    if choice == "gpt-3.5-turbo":
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        st.caption(
            "You have chosen gpt-3.5-turbo. Now upload your csv file.")

    elif choice == "gpt-4":
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        st.caption(
            "You have chosen gpt-4. Now upload your csv file.")

    elif choice == "gpt-4o":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        st.caption(
            "You have chosen gpt-4o. Now upload your csv file.")
    # memory = ConversationBufferMemory()
    # conversation = ConversationChain(
    #     llm=llm, memory=memory, verbose=True)
    st.write("<br>", unsafe_allow_html=True)

    st.subheader("Data Cleaning", divider="rainbow")
    st.write("<br>", unsafe_allow_html=True)


st.write("<br><br>", unsafe_allow_html=True)
st.subheader("Welcome to your personal Generative AI Assistant!")
st.caption("You will now be expertly guided through the steps to start your AI journey into your Data Analytics. Please start by choosing your gpt model in the sidebar.")

csv_file = st.file_uploader("", type="csv")

if csv_file is not None:
    csv_file.seek(0)
    with st.spinner('Processing your dataset... Please wait...'):
        df = pd.read_csv(csv_file, delimiter=';', low_memory=False)

        pandas_agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True
        )

        st.success("Dataset processed successfully!")
        st.write("<br>", unsafe_allow_html=True)

        @st.cache_data
        def primary_steps():
            st.subheader('Starting Exploratory Data Analysis:',
                         divider="rainbow")
            question = 'How many rows and columns are in the dataframe?'
            total_records = pandas_agent.run(question)
            st.caption(total_records)
            missing_values = pandas_agent.run(
                "Are there any missing values in the dataset? If ever there is, respond with 'There are missing values, and they are {mention the missing values}' or 'There are no missing values, that's a very good point.' If ever there are missing values, you can also add 'You can remove the missing values automatically in the Data Cleaning section in the sidebar'.")
            st.caption(missing_values)
            duplicate_values = pandas_agent.run(
                "Are there any rows that have exactly the same information in all their columns in the dataframe? If ever there is, respond with 'There are duplicates, and they are {mention the rows which are duplicates}' or 'There are no duplicate values. Great!' If ever there are duplicate rows, you can also add 'You can remove the duplicates automatically in the Data Cleaning section in the sidebar'.")
            st.caption(duplicate_values)
            st.write('The first 5 rows of the dataset are as follows:')
            st.write(df.head())
            st.write("Here is a list of the column names and their meanings:")
            column_meanings = pandas_agent.run(
                "What are the meanings of the columns? Display this information in bullet points.")
            st.write(column_meanings)
            st.write("Here's a statistical summary of the dataset: ")
            st.write(df.describe())

            return

        primary_steps()

        # def refresh_agent():
        #     global pandas_agent
        #     global df
        #     pandas_agent = create_pandas_dataframe_agent(
        #         llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True)

        st.divider()

        # Creating session states questions and answers
        if "qa_history" not in st.session_state:
            st.session_state.qa_history = []

        def user_question():
            question = st.session_state.new_question
            if question.strip():
                with st.spinner("Processing your question..."):
                    response = pandas_agent.run(
                        question)
                    # refresh_agent()

                    st.session_state.qa_history.append(
                        {"question": question, "answer": response}
                    )
                    st.session_state.new_question = ""

        if st.session_state.qa_history:
            st.write("Previous Interactions:")
            for idx, qa in enumerate(st.session_state.qa_history):
                st.write(f"**Q{idx+1}:** {qa['question']}")
                st.write(f"**A{idx+1}:** {qa['answer']}")
                st.divider()

        st.write(
            "Do you have any additional queries regarding the dataset? ")
        st.text_area(
            "Enter your queries here:",
            key="new_question",
            on_change=user_question,
        )

with st.sidebar:
    with st.expander("Choose your cleaning options"):
        st.write("<br>", unsafe_allow_html=True)

        if st.button("Remove empty cells"):
            df.dropna(inplace=True)
            st.caption("Empty cells removed!")

        elif st.button("Remove duplicate rows"):
            df.drop_duplicates(inplace=True)
            st.caption("Duplicate rows removed!")

        elif st.button("Rename columns"):
            pandas_agent.run(
                "Rename all column names to simplified and understandable names")
            st.write("Column names have been successfully modified.")

        # elif st.button("Remove outliers"):
        #     pass

    st.write("<br>", unsafe_allow_html=True)

    st.subheader("Save your Analysis", divider="rainbow")
    st.write("<br>", unsafe_allow_html=True)

    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1: False}

    def clicked(button):
        st.session_state.clicked[button] = True

    st.button("Save analysis to disk", on_click=clicked, args=[1])
    if st.session_state.clicked[1]:
        st.write("Saving analysis to disk...")
