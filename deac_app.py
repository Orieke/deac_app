import os
import urllib
import requests
import random
from collections import OrderedDict
from IPython.display import display, HTML
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st

from app.embeddings import OpenAIEmbeddings
from app.prompts import STUFF_PROMPT, REFINE_PROMPT, REFINE_QUESTION_PROMPT

# Don't mess with this unless you really know what you are doing
AZURE_SEARCH_API_VERSION = '2021-04-30-Preview'
AZURE_OPENAI_API_VERSION = "2023-03-15-preview"

# Change these below with your own services credentials
AZURE_SEARCH_ENDPOINT = 'https://nksearch.search.windows.net/'
AZURE_SEARCH_KEY = "rZli3K0AQG7nFbdpRRnFdZIn8eTyU5z3YyQ0lePwYPAzSeALTCKk"
AZURE_OPENAI_ENDPOINT = "https://kopenai.openai.azure.com/"
AZURE_OPENAI_API_KEY = "902a52d801a443f3bc2569074c7ea992"

# Setup the Payloads header
headers = {'Content-Type': 'application/json','api-key': AZURE_SEARCH_KEY}

# Index that we are going to query (from Notebook 01 and 02)
index1_name = "cogsrch-py-index"
#index2_name = "cogsrch-index-csv"
indexes = [index1_name]#, index2_name]

st.title("DEAC Smart Reader")
st.subheader(":blue[_You don't need to read :sunglasses:, Just Ask!_]")

#Add session state to track the prompt message 
if 'Question' not in st.session_state:    
    st.session_state['Question'] = "How can I obtain accredication from DEAC?"

#Let's give the question
#QUESTION = "How can I obtain accredication from DEAC?" #"Mention 2 common and 3 severe symptoms of Covid-19?" #"¿Qué son los escenarios empresariales de bots?"#Describe Contoso's existing architecture?"#What is Covid?" 

st.session_state.Question = st.text_area("Type in your question", st.session_state.Question)

if st.button("Get Answer"):    
    with st.spinner("Finding anwser to your question..."):

        # Include code for finding semantic search and open ai response.

        QUESTION = st.session_state.Question 
        
        agg_search_results = []

        for index in indexes:
            url = AZURE_SEARCH_ENDPOINT + '/indexes/'+ index + '/docs'
            url += '?api-version={}'.format(AZURE_SEARCH_API_VERSION)
            url += '&search={}'.format(QUESTION)
            url += '&select=*'
            url += '&$top=10'  # You can change this to anything you need/want
            url += '&queryLanguage=en-us'
            url += '&queryType=semantic' #Note semantic search is not enabled for free tier search service. Answer and Caption are attributes of semantic search
            url += '&semanticConfiguration=my-semantic-config'
            url += '&$count=true'
            url += '&speller=lexicon'
            url += '&answers=extractive|count-3'
            url += '&captions=extractive|highlight-false'

            resp = requests.get(url, headers=headers)
            #print(url)
            #print(resp.status_code)    

            search_results = resp.json()
            agg_search_results.append(search_results)
            #print("Results Found: {}, Results Returned: {}".format(search_results['@odata.count'], len(search_results['value'])))


            #display(HTML('<h4>Top Answers</h4>'))
            #print("Top Answers")
        azureSearchResponse  = ""
        azureOpenaiResponse  = ""
        azureOpenaiSources  = ""
        for search_results in agg_search_results:
            for result in search_results['@search.answers']:
                if result['score'] > 0.5: # Show answers that are at least 50% of the max possible score=1
                    #display(HTML('<h5>' + 'Answer - score: ' + str(result['score']) + '</h5>'))
                    #display(HTML(result['text']))
                    #print('Answer - score: ' + str(result['score']))
                    #print(str(result['text']))
                    d = ""
                    
        #print("\n\n")
        #display(HTML('<h4>Top Results</h4>'))
        azureSearchResponse  += "Top Results" + "\n"

        file_content = OrderedDict()
        content = dict()

        for search_results in agg_search_results:
            for result in search_results['value']:
                if result['@search.rerankerScore'] > 1: # Filter results that are at least 25% of the max possible score=4
                    content[result['id']]={
                                            "title": result['title'],
                                            "chunks": result['pages'],
                                            "language": result['language'], 
                                            "caption": result['@search.captions'][0]['text'],
                                            "score": result['@search.rerankerScore'],
                                            "location": result['metadata_storage_path']                  
                                        }
            
            #After results have been filtered we will Sort and add them as an Ordered list\n",
            for id in sorted(content, key= lambda x: content[x]["score"], reverse=True):
                file_content[id] = content[id]
                #display(HTML('<h5>' + str(file_content[id]['title']) + '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;score: '+ str(file_content[id]['score']) + '</h5>'))
                #display(HTML(file_content[id]['caption']))
                azureSearchResponse  += "Title:" + str(file_content[id]['title']) + "\n"
                azureSearchResponse  += "Score: " + str(file_content[id]['score']) + "\n"
                azureSearchResponse  += "Caption:" + str(file_content[id]['caption']) + "\n"



        #Using Azure OpenAI
        # Set the ENV variables that Langchain needs to connect to Azure OpenAI
        os.environ["OPENAI_API_BASE"] = os.environ["AZURE_OPENAI_ENDPOINT"] = AZURE_OPENAI_ENDPOINT
        os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
        os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"] = AZURE_OPENAI_API_VERSION


        docs = []
        for key,value in file_content.items():
            for page in value["chunks"]:
                docs.append(Document(page_content=page, metadata={"source": value["location"]}))
                
        print("Number of chunks:",len(docs))


        # Select the Embedder model
        if len(docs) < 50:
            # OpenAI models are accurate but slower
            embedder = OpenAIEmbeddings(document_model_name="text-embedding-ada-002", query_model_name="text-embedding-ada-002") 
        else:
            # Bert based models are faster (3x-10x) but not as great in accuracy as OpenAI models
            # Since this repo supports Multiple languages we need to use a multilingual model. 
            # But if English only is the requirement, use "multi-qa-MiniLM-L6-cos-v1"
            # The fastest english model is "all-MiniLM-L12-v2"
            if random.choice(list(file_content.items()))[1]["language"] == "en":
                embedder = HuggingFaceEmbeddings(model_name = 'multi-qa-MiniLM-L6-cos-v1')
            else:
                embedder = HuggingFaceEmbeddings(model_name = 'distiluse-base-multilingual-cased-v2')



        if(len(docs)>1):
            db = FAISS.from_documents(docs, embedder)

            docs_db = db.similarity_search(QUESTION, k=4)


            # Make sure you have the deployment named "gpt-35-turbo" for the model "gpt-35-turbo (0301)". 
            # Use "gpt-4" if you have it available.
            llm = AzureChatOpenAI(deployment_name="kninny-gpt-35-turbo", temperature=0.9, max_tokens=500)
            chain = load_qa_with_sources_chain(llm, chain_type="map_reduce", return_intermediate_steps=True)


            response = chain({"input_documents": docs_db, "question": QUESTION}, return_only_outputs=True)

            answer = response['output_text']

            #display(HTML('<h4>Azure OpenAI ChatGPT Answer:</h4>'))
            #print("Azure OpenAI ChatGPT Answer:")
            azureOpenaiResponse  += answer.split("SOURCES:")[0] + "\n"
            #azureOpenaiResponse  += "Sources:" + "\n"
            for source in answer.split("SOURCES:")[1].replace(" ","").split(","):
                azureOpenaiSources  += str(source) + "\n"
            #azureOpenaiResponse  += answer.split("SOURCES:")[1].replace(" ","").split(",")
        else:
            azureOpenaiResponse  += "No results Found"


         


        #End of code for finding Open AI and sementic search response


        #st.text_area('Azure Search Answer', azureSearchResponse, height=300)
        st.text_area('OpenAI Answer', azureOpenaiResponse, height=300)
        st.text_area('Sources', azureOpenaiSources, height=200)
        st.success('Done!')


