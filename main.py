import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from langchain.prompts import MessagesPlaceholder
from langsmith import Client
from Utils.utils import load_documents, get_faiss_vectorStore
from dotenv import load_dotenv

client = Client()
load_dotenv()

st.set_page_config(page_title="Chatbot ARINC 653P1-2", page_icon=":robot_face:")
st.title("Chatbot ARINC 653P1-2")

with st.sidebar:
    st.title("LLM ARINC 653P1-2")
    st.markdown(''' 
    ## ARINC 653P1-2
    This app is for the LLM ARINC 653P1-2.
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    
    ''')
    add_vertical_space(2)
    st.write("Follow me on [Linkedin](https://www.linkedin.com/in/abhigaelcarranza/)")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

@st.cache_resource(ttl="1h")
def configure_retriever():
    # documents=load_documents()
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store=get_faiss_vectorStore(embeddings)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

tool= create_retriever_tool(
    configure_retriever(),
    "search_arinc_653P1-2_docs",
    "Searches and returns documents regarding ARINC 653P1-2. so if you are ever asked about ARINC 653P1-2 you should use this tool."
    )

tools=[tool]
llm=ChatOpenAI(temperature=0,streaming=True,model_name="gpt-3.5-turbo",openai_api_key=openai_api_key,max_tokens=1000)
message=SystemMessage(
    content=(
        """
        You are an expert chatbot specializing in the ARINC 653P1-2 standard. Your knowledge is confined to this specific area. 
        When engaging in a conversation, always assume that the questions are related to ARINC 653P1-2 . If any ambiguity arises, 
        default to interpreting questions in the context of this subject. If a question is posed outside the realm of ARINC 653P1-2, 
        politely inform the inquirer that your expertise is limited to this domain, and you may not be able to provide an accurate 
        response to unrelated topics.
        """
    )
)

prompt=OpenAIFunctionsAgent.create_prompt(
    system_message=message,
    extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
)
agent=OpenAIFunctionsAgent(llm=llm,tools=tools,prompt=prompt)
agent_executor=AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
)

memory=AgentTokenBufferMemory(llm=llm,max_token_limit=4000)
starter_message="Ask me a question about ARINC 653P1-2."

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"]=[AIMessage(content=starter_message)]

def send_feedback(run_id,score):
    client.create_feedback(run_id,"user_score",score=score)
    
for msg in st.session_state.messages:
    if isinstance(msg,AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg,HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)

if prompt:=st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback=StreamlitCallbackHandler(st.container())
        response=agent_executor(
            {"input":prompt, "history":st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))
        st.write(response["output"])
        memory.save_context({"input":prompt},response)
        st.session_state['messages']=memory.buffer
        run_id=response["__run"].run_id
        
        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
        with col_text:
            st.text("Feedback:")

        with col1:
            st.button("👍", on_click=send_feedback, args=(run_id, 1))

        with col2:
            st.button("👎", on_click=send_feedback, args=(run_id, 0))