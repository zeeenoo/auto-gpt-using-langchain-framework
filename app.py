import os 
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
os.environ['OPENAI_API_KEY'] = apikey


#app framework
st.title('Youtube Title Generator 🩴')
prompt = st.text_input('enter the subject')

#prompt template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template= 'write me a youtube video about {topic}'
)

#LLM 
llm = OpenAI(temperature=0.9)
title_chain = LLMChain (llm=llm , prompt=title_template, verbose=True)
#show smth on the screen IF there's a prompt

if prompt : 
    # response = llm(prompt)
    response = title_chain.run(topic = prompt, )
    st.write(response)


