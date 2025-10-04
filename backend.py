from langgraph.graph import StateGraph,START,END
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    message:Annotated[list[BaseMessage],add_messages]
llm=ChatOpenAI()
def chat_node(state:ChatState):
    message=state["message"]
    response=llm.invoke(message)
    return {"message":[response]}


graph=StateGraph(ChatState)
checkpointer=MemorySaver()
graph.add_node("chat_node",chat_node)
graph.add_edge(START,chat_node)
graph.add_edge(chat_node,END)
chatbot=graph.compile(checkpointer=checkpointer)
initial_state={
    "message":[HumanMessage(content="What is the capital of pakistan")]
}
thread="1"
while True:
    user_message=input("type Message: ")
    print("user_message",user_message)
    if user_message.strip().lower() in ["exit","quit","bye"]:
        break
    config={"configuration":{"thread":thread}}
    response=chatbot.invoke({"message":[HumanMessage(content=user_message)]},config=config)
    print("AI Message",response["message"][-1].content)