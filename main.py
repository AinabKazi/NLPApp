import streamlit as st
from time import sleep
from stqdm import stqdm # for getting animation after submit event
import  pandas as pd
from transformers import pipeline
import json
import spacy
import spacy_streamlit


def draw_all(
        key,
        plot=False,
):
    st.write(
        """
        # NLP Web App

        This Natural Language Processing Based Web App can do anything u can imagine with Text. 😱 

        This App is built using pretrained transformers which are capable of doing wonders with the Textual data.

        ```python
        # Key Features of this App.
        1. Advanced Text Summarizer
        2. Named Entity Recognition
        3. Sentiment Analysis
        4. Question Answering
        5. Text Completion

        ```
        """
    )


with st.sidebar:
    draw_all("sidebar")


def main():
    st.title("NLP Web App")
    menu = ["--Select--", "Summarizer", "Named Entity Recognition",
            "Sentiment Analysis", "Question Answering", "Text Completion"]
    choice = st.sidebar.selectbox("Choose What u wanna do !!", menu)

    if choice == "--Select--":
        st.write("""

                 This is a Natural Language Processing Based Web App that can do   
                 anything u can imagine with the Text.
        """)

        st.write("""

                Natural Language Processing (NLP) is a computational technique
                to understand the human language in the way they spoke and write.
        """)

        st.write("""

                 NLP is a sub field of Artificial Intelligence (AI) to understand
                 the context of text just like humans.
        """)

        st.image('chatbot.jpg')




    elif choice == "Summarizer":
        st.subheader("Text Summarization")
        st.write(" Enter the Text you want to summarize !")
        raw_text = st.text_area("Your Text", "Enter Your Text Here")
        num_words = st.number_input("Enter Number of Words in Summary")

        if raw_text != "" and num_words is not None:
            num_words = int(num_words)
            summarizer = pipeline('summarization')
            summary = summarizer(raw_text, min_length=num_words, max_length=50)
            s1 = json.dumps(summary[0])
            d2 = json.loads(s1)
            result_summary = d2['summary_text']
            result_summary = '. '.join(list(map(lambda x: x.strip().capitalize(),
                                                result_summary.split('.'))))
            st.write(f"Here's your Summary : {result_summary}")
    elif choice=="Named Entity Recognition":
        nlp = spacy.load("en_core_web_trf")
        st.subheader("Text Based Named Entity Recognition")
        st.write(" Enter the Text below To extract Named Entities !")

        raw_text = st.text_area("Your Text","Enter Text Here")
        if raw_text !="Enter Text Here":
            doc = nlp(raw_text)
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe("ner").labels, title= "List of Entities")
    elif choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")
        sentiment_analysis = pipeline("sentiment-analysis")
        st.write(" Enter the Text below To find out its Sentiment !")

        raw_text = st.text_area("Your Text", "Enter Text Here")
        if raw_text != "Enter Text Here":
            result = sentiment_analysis(raw_text)[0]
            sentiment = result['label']
            for _ in stqdm(range(50), desc="Please wait a bit. The model is fetching the results !!"):
                sleep(0.1)
            if sentiment == "POSITIVE":
                st.write("""# This text has a Positive Sentiment.  🤗""")
            elif sentiment == "NEGATIVE":
                st.write("""# This text has a Negative Sentiment. 😤""")
            elif sentiment == "NEUTRAL":
                st.write("""# This text seems Neutral ... 😐""")
if __name__ == '__main__':
    main()