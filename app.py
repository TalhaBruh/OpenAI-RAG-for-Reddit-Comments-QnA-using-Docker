import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from reddit_utils import is_reddit_url
import scraper
import config
import openai


st.set_page_config(
    page_title="RedditGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("RedditGPT")

if "messages" not in st.session_state.keys():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello there! Enter a reddit URL and ask away!"}
    ]


prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a helpful assistant. The user will provide contents of reddit posts and ask questions
    about them. Your job is to summarize them when the user provides the contents and answer 
    
    chat_history: {chat_history}

    Human: {question}

    AI:"""
)

# openai.api_key = config.openai_api_key

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    openai_api_key= config.openai_api_key
    )

memory= ConversationBufferWindowMemory(memory_key="chat_history", k=5 )

llm_chain = LLMChain(
    llm = llm,
    memory= memory,
    prompt=prompt
)


for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_prompt = st.chat_input()

reddit_content= None
if user_prompt is not None:
    with st.chat_message("user"):
        st.write(user_prompt)
    if is_reddit_url(user_prompt): 
        with st.chat_message("assistant"): 
            with st.spinner("Reading the Reddit post..."):      
                messages, vectorstore = scraper.generate_prompt_for_thread(user_prompt)
                st.session_state['vectorstore'] = vectorstore
                reddit_content = messages[0]["content"]
            st.write("Done reading!")

    user_prompt = reddit_content if reddit_content is not None else user_prompt
    st.session_state.messages.append(
        {"role" : "user", "content": user_prompt}
    )


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            if "vectorstore" in st.session_state.keys():
                vectorstore = st.session_state["vectorstore"]
                rqa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
                ai_response = rqa({"question": user_prompt})["answer"]
            else:
                ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)
