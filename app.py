import streamlit as st
import pvleopard
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from pydub import AudioSegment
import matplotlib.pyplot as plt
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize Picovoice Leopard
leopard = pvleopard.create(access_key=os.getenv("ACCESS_KEY"))

# Load a pre-trained transformer model from Hugging Face for classification
model_name = "facebook/bart-large-mnli"  # You can choose any model suitable for text evaluation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Initialize the Groq model
chat = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

# Initialize the output parser
parser = StrOutputParser()

# Define the system prompt
system_prompt = """Analyze the provided transcription for Coherence, Fluency, Pronunciation, Lexical Resource, and Grammatical Range and Accuracy. Provide detailed feedback on each aspect to improve the overall communication."""

# Define the human prompt with placeholders for context and scores
human_prompt = """The transcription is: {context}\n\nThe scores are:\nCoherence: {coherence_score}\nFluency: {fluency_score}\nPronunciation: {pronunciation_score}\nLexical Resource: {lexical_resource_score}\nGrammatical Range and Accuracy: {grammar_score}"""

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])

# Function to transcribe audio
def transcribe_audio(audio_file):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_wav_path = "temp.wav"
    audio.export(temp_wav_path, format="wav")
    transcription, _ = leopard.process_file(temp_wav_path)
    return transcription

# Function for parameter analysis
def parameter_analysis(transcription):
    classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer)
    labels = ["Fluency", "Coherence", "Lexical Resource", "Grammatical Range and Accuracy", "Pronunciation"]
    results = classifier(transcription, candidate_labels=labels)
    analysis_scores = {label: score * 10 for label, score in zip(results['labels'], results['scores'])}
    speaking_score = sum(analysis_scores.values()) / len(analysis_scores)
    analysis_scores["Speaking Score"] = speaking_score
    return analysis_scores

# Function to provide feedback based on the scores
def provide_feedback(analysis_scores):
    feedback = {}
    for parameter, score in analysis_scores.items():
        if score >= 8:
            feedback[parameter] = "Excellent! Keep up the great work."
        elif score >= 5:
            feedback[parameter] = "Good, but there's room for improvement. Practice more and pay attention to this area."
        else:
            feedback[parameter] = "Needs significant improvement. Consider focusing on this aspect to enhance your speaking skills."
    return feedback

# Function to visualize scores
def visualize_scores(analysis_scores):
    labels = list(analysis_scores.keys())
    scores = list(analysis_scores.values())
    fig, ax = plt.subplots()
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlim(0, 10)
    ax.set_xlabel('Scores')
    ax.set_title('Speech Analysis Scores')
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Verbius: Speech Analysis Tool")
    st.sidebar.title("Upload Audio File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=['mp3', 'wav'])

    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav', start_time=0)
        transcription = transcribe_audio(uploaded_file)
        st.write("Transcription:", transcription)

        analysis_scores = parameter_analysis(transcription)
        st.write("Parameter Analysis:", analysis_scores)

        feedback = provide_feedback(analysis_scores)
        st.write("Feedback:")
        for parameter, advice in feedback.items():
            st.write(f"{parameter}: {advice}")

        visualize_scores(analysis_scores)

        # Invoke the chain for additional feedback
        output = prompt | chat | parser | {"context": transcription, "coherence_score": analysis_scores["Coherence"], "fluency_score": analysis_scores["Fluency"], "pronunciation_score": analysis_scores["Pronunciation"], "lexical_resource_score": analysis_scores["Lexical Resource"], "grammar_score": analysis_scores["Grammatical Range and Accuracy"]}
        st.write("Additional Feedback:", output)

if __name__ == "__main__":
    main()
