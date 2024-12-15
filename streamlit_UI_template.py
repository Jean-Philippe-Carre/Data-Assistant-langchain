
import streamlit as st

# 1 asterisk - italic
# 2 asterisks - bold
# 3 asterisks - bold italic

st.title("Welcome to this Generative AI assistant! 🤖")
# st.header("")
st.subheader("Harness the power of LLms at your fingertips.")
st.caption("You have the possibility to upload your dataset and AI will help you find meaningful insights into your data. Enjoy!")

with st.sidebar:
    st.write('*This is a sidebar.Please **upload** your csv file here:*')
    st.divider()

    st.caption('''In case you didn't know, this is the space where an input field should be placed so that you can upload your precious data set. Once it is uploaded, the AI will automatically clean the dataset and then you'll be able to continue.
''')
    st.divider()
    st.write("<p style = 'text-align:center'>✨ AI Analytics ✨</p>",
             unsafe_allow_html=True)
    st.divider()

    # use session state for buttons to remain True as they are stateless
    # initialise the key in session state
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1: False}

    # function to update the value in session state

    def clicked(button):
        st.session_state.clicked[button] = True

    st.button("Step into the Future", on_click=clicked, args=[1])
    if st.session_state.clicked[1]:
        st.subheader("Here we go!")
        st.write("Why don't we upload a dataset to begin with, shall we...")
        user_csv = st.file_uploader("Upload your Dataset here", type="csv")

    choice = st.radio(
        label="Choose your preferred LLM:",
        options=("Choose your preferred LLM:",
                 "gpt-3.5-turbo", "gpt-4", "gpt-4o"),
    )
    if choice == "gpt-3.5-turbo":
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        st.caption(
            "You have chosen gpt-3.5-turbo. Time to upload your csv file...")

with st.expander("This is a cool expander"):
    st.write("Option 1")
    st.write("Option 2")
    st.write("Option 3")
