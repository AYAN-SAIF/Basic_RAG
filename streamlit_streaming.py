import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])
user_input=st.chat_input("Type here")


if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})
    with st.chat_message("user"):
        st.text(user_input)
    thread="1"
    config={"configuration":{"thread":thread}}
    # response=chatbot.invoke({"message":[HumanMessage(content=user_input)]},config=config)
    # ai_message=response["message"][-1].content

    
    with st.chat_message("assistant"):
        ai_message=st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {"message":[HumanMessage(content=user_input)]},
                config={"configuration":{"thread":thread}},
                stream_mode="messages"
            )
        )
        st.session_state["message_history"].append({"role":"assistant","content":ai_message})
        