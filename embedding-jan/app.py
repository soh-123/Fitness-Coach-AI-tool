from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import os
import streamlit as st


api_key = os.environ.get("OPENAI_API_KEY")

loader = CSVLoader(file_path="dataset-generation/dataset.csv")
document = loader.load()


embeddings = OpenAIEmbeddings()
db = faiss.FAISS.from_documents(document, embeddings)

def retrieve_info(querry):
    similar_response = db.similarity_search(querry, k=3)
    page_content_array = [doc.page_content for doc in similar_response]
    print(page_content_array)
    return page_content_array

llm = ChatOpenAI(model="gpt-4-1106-preview")
template = """
    You are a professional fitness coach. You had too many clients who lost weight effectively and became fit by using your advises and following your plans. 
    I will share the user's personal details like weight, height, gender, age and target goal and you will create a detailed plan that suites my his/her life style, it should follow all the following details:
    1/ Respond should be detailed and simmilar to past best practices, in terms of tone of voice, list and other details.
    2/ if the best practices is irrelevant, then try to mimic the style of the best practice

    Below is the the user enquerry:
    {message}
    Here is a list of best practices of how you should respond in similar sceinario:
    {best_practice}

    Please respond with the best weight lose plan that is easy to be followed.
"""
prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template= template
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice= best_practice)
    return response


def main():
    st.set_page_config(
        page_title="Weight lose plan generator",
        page_icon="💪🏻"
    )
    st.header("Weight lose plan generator 💪🏻")
    message = st.text_area("User message")

    if message:
        st.write("Generating your fitness plan ..")
        result = generate_response(message)
        st.info(result)

if __name__ == "__main__":
    main()



# message = "I am a 50 years female, weight is 75kg and height is 155cm, create a weight lose plan for 6 months to reach my target weight"