# import os
# import warnings
# warnings.filterwarnings("ignore")

# from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# # .env file se Gemini API key load karna
# load_dotenv()

# # Paths
# DB_DIR = "./vector_store"

# def setup_rag_pipeline():
#     print("🧠 AI Brain load ho raha hai...")

#     # 1. Database Retriever
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectorstore = Chroma(
#         collection_name="mindgigs_collection",
#         embedding_function=embeddings,
#         persist_directory=DB_DIR
#     )
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#     # 2. LLM (Gemini) Setup
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

#     # 3. Prompt Engineering
#     system_prompt = (
#         "Tum 'MindGigs' ke official aur professional AI Assistant ho. "
#         "Tumhara maqsad website visitors ko guide karna aur unke sawalat ke jawabat dena hai. "
#         "Neeche diye gaye context ko parh kar user ke sawal ka jawab do. "
#         "Agar context mein jawab mojood nahi hai, toh politely bata do ke 'Mujhe iski maloomat nahi hain, baraye meharbani mindgigspk@gmail.com par contact karein'. "
#         "Khud se koi maloomat (hallucination) mat banana. Jawab clear aur friendly tone mein dena.\n\n"
#         "Context: {context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}")
#     ])

#     # 4. Modern LCEL Pipeline (Bina .chains ke)
#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     rag_chain = (
#         {"context": retriever | format_docs, "input": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     return rag_chain

# # Chat Loop
# if __name__ == "__main__":
#     if "GOOGLE_API_KEY" not in os.environ:
#         print("❌ Error: .env file mein GOOGLE_API_KEY nahi mili!")
#     else:
#         chatbot = setup_rag_pipeline()
#         print("\n✅ MindGigs Bot Ready hai! (Chat khatam karne ke liye 'exit' likhein)\n")
#         print("-" * 50)
        
#         while True:
#             user_query = input("Aapka Sawal: ")
#             if user_query.lower() in ['exit', 'quit']:
#                 print("Bot: Khuda Hafiz!")
#                 break
                
#             # AI se jawab mangna (Direct query pass ki jati hai LCEL mein)
#             response = chatbot.invoke(user_query)
#             print(f"Bot: {response}\n")
#             print("-" * 50)

import os
import warnings
import uuid
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables (.env)
load_dotenv()

# Paths
FILE_PATH = "data/mindgigs_data.txt"
DB_DIR = "./vector_store"

def build_vector_database(embeddings):
    """Yeh function sirf tab chalega jab vector_store folder nahi hoga"""
    print("🚀 Building Memory Bank Automatically...")
    
    loader = TextLoader(FILE_PATH)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = Chroma(
        collection_name="mindgigs_collection",
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )
    
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)
    print("✅ Database created successfully!")
    
    return vectorstore

def setup_rag_pipeline():
    """Main pipeline jo app.py call karega"""
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Auto-Check: Agar Database folder nahi hai, toh pehle usay banao
    if not os.path.exists(DB_DIR):
        if not os.path.exists(FILE_PATH):
            raise FileNotFoundError(f"❌ Error: Text file '{FILE_PATH}' missing. Please add it.")
        vectorstore = build_vector_database(embeddings)
    else:
        # Agar folder maujood hai, toh seedha load karo (Time bachega)
        vectorstore = Chroma(
            collection_name="mindgigs_collection",
            embedding_function=embeddings,
            persist_directory=DB_DIR
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Fast Lite Model for better speed
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

    system_prompt = (
       "You are the official and professional AI Assistant of 'MindGigs'. "
        "Your purpose is to guide website visitors and answer their questions. "
        "Read the context below and answer the user's question. "
        "If the answer is not present in the context, politely state: 'I do not have information regarding this, please contact mindgigspk@gmail.com'. "
        "Do not hallucinate or make up any information on your own. Provide the answer in a clear and friendly tone.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
