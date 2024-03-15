import os
import streamlit as st
import openai as client
import time

# Streamlit 앱 정의
def main():
    client.api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = "asst_KGi6fyaWy277PKOCJuPiyaIy"
    assitant = client.beta.assistants.retrieve(assistant_id)

    st.title("Multi-turn Chatbot with Streamlit and OpenAI Assistant with retrieval")
    st.chat_input(placeholder="대화를 입력해주세요.", key="chat_input")

    thread = None
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state["thread_id"] = thread.id
    else:
        thread_id = st.session_state["thread_id"]
        thread = client.beta.threads.retrieve(thread_id)

    thread_messages = client.beta.threads.messages.list(thread.id, order="asc")
    for message in thread_messages.data:
        with st.chat_message(message.role):
            st.markdown(message.content[0].text.value)

    if user_input := st.session_state["chat_input"]:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        user_message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )        

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        while run.status != "completed" :
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            time.sleep(0.1)

        with st.chat_message("assistant"):
            thread_messages = client.beta.threads.messages.list(thread.id, order="asc")
            message = thread_messages.data[-1].content[0].text.value
            st.markdown(message)

if __name__ == "__main__":
    main()
