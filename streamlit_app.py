import os
import streamlit as st
import faiss
import time

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from loguru import logger

load_dotenv()
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
WOKRING_PROJECT_DIR = os.getenv("WORKING_PROJECT_DIR")
DEFAULT_SYSTEM_PROMPT = (
    'You are a helpful AI assistant of Sayid Muhammad Heykal. You may refine the query if needed.',
    'You are going to answer all the answer based on provided document that accessable from the tool function',
    'You have access to a tool retrieve content from a document. ',
    'Always looking first to the tool to help answer user query...'
)
DOC_PATH = os.path.join(WOKRING_PROJECT_DIR, "docs", "resume.pdf")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # macos env issue
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ------------ SIMPLE AGENT ------------

class BuildAgent():
    def __init__(self, system_prompt: str = None):
        self._system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
        )

        self.embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
        )


    def build(self):
        vector_store = self.docs_to_vector_store()

        @tool(response_format='content_and_artifact')
        def retrieve_content(query: str):
            '''Retrieve the information to help answer query

            Args:
                query (str): Prompt message from input user
            '''
            retrieved_docs = vector_store.similarity_search(
                query=query,
                k=5,
            )

            serialized = "\n\n".join(
                (f"metadata: {doc.metadata}\ncontent: {doc.page_content}")
                for doc in retrieved_docs
            )

            return serialized, retrieved_docs

        return create_agent(
            self.model, 
            tools=[retrieve_content],
            system_prompt=self._system_prompt
        )
    
    def docs_to_vector_store(self, file_path: str = None):
        if not file_path:
            file_path = "/Users/heykalsayid/Desktop/chatbot-webinar/rag_langchain_streamlit/docs/resume.md"

        with st.spinner("Load necessary documents..."):
            loader = UnstructuredMarkdownLoader(
                file_path,
                mode="single",
                strategy="fast",
            )
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                add_start_index=True,
            )
            all_splits = text_splitter.split_documents(docs)
            all_splits = [d for d in all_splits if d.page_content.strip()]

            if not all_splits:
                raise ValueError("No valid text extracted from PDF")

            vector_store = FAISS.from_documents(
                all_splits,
                embedding=self.embedding
            )

            doc_ids = list(vector_store.index_to_docstore_id.values())

            logger.success(f"Added {len(doc_ids)} documents to vector store")

        return vector_store



# ------------ INTERFACE CODE ------------

st.title("Personal Chatbot 💬")
st.caption("Want to chat me in person? Find me at the coffee shop on Main Street! ☕️")

print(st.session_state)

if 'agent' not in st.session_state:
    st.session_state.agent = BuildAgent().build()

# Setup session state to store messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            'role': 'assistant',
            'content': 'Hello! I am your personal chatbot. How can I assist you today?',
        }
    ]

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


# Prompt input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.chat_history.append(
        {
            'role': 'user',
            'content': prompt,
        }
    )
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        logger.info("Generating response...")
        with st.spinner("Generating response..."):
            try:
                message_placeholder = st.empty()
                full_response = ""
                response = st.session_state.agent.invoke({"messages" : [
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ]})
                response = response['messages'][-1].content[0]['text']
                
                # Simulate stream response
                for chunk in response.split(" "):
                    full_response += chunk + " "
                    time.sleep(0.005)
                    message_placeholder.markdown(full_response + "✏️ ")
                message_placeholder.markdown(full_response)
                
                # save full response to chat history
                st.session_state.chat_history.append(
                    {
                        'role': 'assistant',
                        'content': full_response,
                    }
                )

            except Exception as e:
                print(f"Something error -> {e}")
                st.error(e)
        
        logger.success("Response generated")
        
        
    