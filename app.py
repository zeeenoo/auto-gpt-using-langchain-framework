import os 
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey


#app framework
st.title('Youtube Title Generator 🩴')
prompt = st.text_input('enter the subject')

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template= 'write me a youtube video about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title','wikipedia_research'],
    template= 'write me a youtube video script base on this title :  {title} by leveraging this wikipedia research : {wikipedia_research}'
)

# Memory :for storing the history
title_memory = ConversationBufferMemory(input_key='topic' , memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title' , memory_key='chat_history')




#LLM 
llm = OpenAI(temperature=0.9)
title_chain = LLMChain (llm=llm , prompt=title_template, verbose=True, output_key='title',memory=title_memory)
script_chain = LLMChain (llm=llm , prompt=script_template, verbose=True, output_key='script', memory=script_memory)

#here we want to concatinate the chains 
wiki = WikipediaAPIWrapper()
# sequential_chain = SequentialChain(chains=[title_chain,script_chain], input_variables=['topic'], output_variables=['title','script'], verbose=True)


#show smth on the screen IF there's a prompt

if prompt : 
    # response = llm(prompt)
    # response = sequential_chain.run( {'topic': prompt} )
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    response = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(response)


    with st.expander('Title History'):
        st.info(title_memory.buffer) # like accordion expander

    with st.expander('Message History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia research'):
        st.info(wiki_research)

