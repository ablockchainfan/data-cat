"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from ingest_data import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
import os

import glob

dirName = '.'

files = [f for f in glob.glob(dirName + "\*.pkl", recursive=False)]
files = [item.replace(".pkl", "") for item in files]
# def load_chain():
#     """Logic for loading the chain you want to use should go here."""
#     llm = OpenAI(temperature=0)
#     chain = ConversationChain(llm=llm)
#     return chain

print(files)

# From here down is all the StreamLit UI.
st.set_page_config(page_title="Chat with your Docs", page_icon=":robot:")
st.header("Chat with your Docs")

uploaded_file = st.file_uploader("Upload a document you would like to chat about", type=None, accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible")

# check if file is uploaded and file does not exist in data folder
if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
    # write the file to data directory
    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully")
    with st.spinner('Document is being vectorized...'):
        embed_doc()
# open vectorstore.pkl if it exists in current directory
# option = st.selectbox(
#     'Which DB you want to talk with ?',
#     files)

# st.write('You selected:', option)

if "vectorstore.pkl" in os.listdir("."):
    with open("vectorstore.pkl", "rb") as f:
        vectorstore = pickle.load(f)
        print("Loading vectorstore...")

    chain = get_chain(vectorstore)

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

placeholder = st.empty()


input_text = st.text_input("You: ", value="",  key="input")
user_input = input_text
print(st.session_state.input)

print(user_input)

if user_input:
    docs = vectorstore.similarity_search(user_input)
    # if checkbox is checked, print docs
    # st.session_state["input"] = ""
    print(len(docs))
    # output = chain.run(input=user_input, vectorstore = vectorstore, context=docs[:2], chat_history = [], question= user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)
    output = "result"

    st.session_state.past.append(user_input)
    print(st.session_state.past)
    st.session_state.generated.append(output)
    print(st.session_state.past)

    if st.checkbox("Show similar documents"):
        st.markdown(docs)
    

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
    
    