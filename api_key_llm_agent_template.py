from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import find_dotenv, load_dotenv
from apikey import apikey

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.agents.agent_toolkit import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.utilities import WikipediaAPIWrapper

import streamlit as st

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


# "openai_key" is the variable in which i declared the key in .env
apikey = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature=0)

with st.expander("What is Generative AI"):
    st.write(llm("What is Generative AI"))

# df = pd.read_csv(user_csv, low_memory=False)

# df is the variable in which we stored the pandas dataframe
pandas_agent = create_pandas_dataframe_agent(
    llm, df, verbose=True, allow_dangerous_code=True)
