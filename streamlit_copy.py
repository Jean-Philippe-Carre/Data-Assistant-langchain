import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time

from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

with open("styles.css") as css:
    st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)

# ----------------------- SIDEBAR LLM Radio Buttons ----------------------

with st.sidebar:
    st.write("<br><br>", unsafe_allow_html=True)
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

    st.write("<br>", unsafe_allow_html=True)

# ----------------------- Intro & File Uploader -----------------------

st.write("<br>", unsafe_allow_html=True)

st.markdown(
    """:rainbow[Welcome to your personal Generative AI Assistant]""")

intro = "You will now be guided through the steps to start your AI journey into your Data Analytics. Please start by choosing your gpt model in the sidebar."
st.write(intro)

csv_file = st.file_uploader("", type="csv")

# ----------------------- Dataframe & Pandas Agent creation -----------------------

if csv_file is not None:
    df = pd.read_csv(csv_file, delimiter=';', low_memory=False)

    pandas_agent = create_pandas_dataframe_agent(
        llm, df, verbose=True, handle_parsing_errors=True, allow_dangerous_code=True
    )

    st.write("<br>", unsafe_allow_html=True)

    st.divider()

# -------------------- Starting Exploratory Data Analysis Function --------------------

    @st.cache_data
    def Starting_Exploratory_Data_Analysis():
        st.subheader('Starting Exploratory Data Analysis:',
                     divider="rainbow")

        question = 'How many rows and columns are in the dataframe?'
        total_records = pandas_agent.run(question)

        def ttl_records():
            for word in total_records.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(ttl_records())

        missing_values = pandas_agent.run(
            "Are there any missing values in the dataset? If ever there is, respond with 'There are missing values, and they are {mention the missing values}' or 'There are no missing values, that's a very good point.' If ever there are missing values, you can also add 'You can remove the missing values automatically in the Data Cleaning section in the sidebar'.")

        def missing_val():
            for word in missing_values.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(missing_val())

        duplicate_values = pandas_agent.run(
            "Are there any rows that have exactly the same information in all their columns in the dataframe? If ever there is, respond with 'There are duplicates, and they are {mention the rows which are duplicates}' or 'There are no duplicate values. Great!' If ever there are duplicate rows, you can also add 'You can remove the duplicates automatically in the Data Cleaning section in the sidebar'.")

        def duplicate_val():
            for word in duplicate_values.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(duplicate_val())

        head = 'The first 5 rows of the dataset are as follows:'

        def dfhead():
            for word in head.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(dfhead())

        st.write(df.head())

        col_names = "Here is a list of the column names and their meanings:"

        def col():
            for word in col_names.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(col())

        column_meanings = pandas_agent.run(
            "What are the meanings of the columns? Display this information in bullet points.")

        def col_meanings():
            for word in column_meanings.split(" "):
                yield word + " "
                time.sleep(0.06)
        st.write(col_meanings())

        st.write("Here's a statistical summary of the dataset: ")
        st.write(df.describe())

        return

    Starting_Exploratory_Data_Analysis()

    st.divider()

# ------------------ Initialise Session State for Conversation History ------------------

    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

# ----------------------- User Question Function -----------------------

    def user_question():
        question = st.session_state.new_question
        if question.strip():
            with st.spinner("Processing your question..."):
                try:
                    response = pandas_agent.run(question)
                except ValueError as e:
                    if "Could not parse LLM output" in str(e):
                        response = "There was an error understanding the output. Please re-phrase your question."
                        st.error(response)
                    else:
                        raise e

            st.session_state.qa_history.append(
                {"question": question, "answer": response})
            st.session_state.new_question = ""

    if st.session_state.qa_history:
        st.write("Previous Interactions:")
        for idx, qa in enumerate(st.session_state.qa_history):
            st.write(f"**Q{idx+1}:** {qa['question']}")
            st.write(f"**A{idx+1}:** {qa['answer']}")
            st.divider()

    st.write("Do you have any additional queries regarding the dataset? ")
    st.text_area(
        "Enter your queries here:",
        key="new_question",
        on_change=user_question,
    )

# ------------------- SIDEBAR Cleaning & Machine Learning Buttons -------------------


with st.sidebar:
    st.subheader("Data Cleaning", divider="rainbow")
    st.write("<br>", unsafe_allow_html=True)

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

        elif st.button("Remove outliers"):
            st.caption("Remove Outliers will be implemented in the next update")
    st.write("<br>", unsafe_allow_html=True)

    st.subheader("Machine Learning", divider="rainbow")
    st.write("<br>", unsafe_allow_html=True)

    with st.expander("Choose your Machine Learning Algorythms"):
        st.caption(
            "Machine Learning algorythms are being developed and will be implemented soon...")
    st.write("<br>", unsafe_allow_html=True)

    st.subheader("Save Analysis to disk", divider="rainbow")
    st.write("<br>", unsafe_allow_html=True)

    with st.expander("Instructions to save your insights as pdf"):
        st.write("""
                    \n(1) First close the sidebar
                    \n(2) Then click on the three vertical dots on the top right corner \nof the page
                    \n(3) Click Print
                    \n(4) Choose Save as PDF
                    \n(5) Save the file to your desired location
                 """)
