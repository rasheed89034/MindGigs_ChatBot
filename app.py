# import streamlit as st
# from rag_engine import setup_rag_pipeline

# # 1. Web Page ki Settings
# st.set_page_config(page_title="MindGigs AI", page_icon="🤖", layout="centered")

# st.title("🤖 MindGigs AI Assistant")
# st.write("MindGigs ki services aur courses ke baare mein koi bhi sawal poochein!")

# # 2. AI Brain ko Load Karna (Taake website fast chale)
# @st.cache_resource
# def load_bot():
#     return setup_rag_pipeline()

# chatbot = load_bot()

# # 3. Chat History Memory (Purane messages save rakhne ke liye)
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Purane messages ko screen par dikhana
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # 4. User Input aur AI Jawab
# user_input = st.chat_input("Apna sawal yahan likhein (jaise: kon kon sy course available hai?)...")

# if user_input:
#     # User ka sawal screen par lagana
#     st.chat_message("user").markdown(user_input)
#     st.session_state.messages.append({"role": "user", "content": user_input})

#     # AI se jawab mangwana
#     with st.spinner("AI soch raha hai..."):
#         ai_response = chatbot.invoke(user_input)
        
#     # AI ka jawab screen par lagana
#     with st.chat_message("assistant"):
#         st.markdown(ai_response)
#     st.session_state.messages.append({"role": "assistant", "content": ai_response})

import streamlit as st
from rag_engine import setup_rag_pipeline

# 1. Web Page Settings
st.set_page_config(page_title="MindGigs AI", page_icon="🤖", layout="centered")

st.title("🤖 MindGigs AI Assistant")
st.write("Ask about MindGigs services and Courses!")

# 2. Loading AI Brain
@st.cache_resource
def load_bot():
    return setup_rag_pipeline()

chatbot = load_bot()

# 3. Chat History Memory (Purane messages save rakhne ke liye)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Purane messages ko screen par dikhana
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. User Input aur AI Jawab
user_input = st.chat_input("Write your query here...")

if user_input:
    # Bring User query on screen 
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AI answer (Streaming Mode - No loading spinner needed)
    with st.chat_message("assistant"):
        # .stream() use karne se AI ka jawab real-time type hoga
        response_stream = chatbot.stream(user_input)
        
        # st.write_stream() automatically chunks ko typewriter effect mein dikhata hai
        ai_response = st.write_stream(response_stream)
        
    # Pura jawab save karna memory ke liye
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
