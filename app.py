import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pypdf import PdfReader
from happytransformer import HappyTextToText, TTSettings

fake_news_tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
fake_news_model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

essay_tokenizer = AutoTokenizer.from_pretrained("Kevintu/Engessay_grading_ML")
essay_model = AutoModelForSequenceClassification.from_pretrained("Kevintu/Engessay_grading_ML")

grammar_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
grammar_args = TTSettings(num_beams=5, min_length=1)

def predict_fake(title, text):
    input_str = "<title>" + title + "<content>" + text + "<end>"
    input_ids = fake_news_tokenizer.encode_plus(input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    device = 'cpu'
    fake_news_model.to(device)
    with torch.no_grad():
        output = fake_news_model(input_ids["input_ids"].to(device), attention_mask=input_ids["attention_mask"].to(device))
    probs = torch.nn.Softmax(dim=1)(output.logits)[0]
    return dict(zip(["Fake", "Real"], [x.item() for x in probs]))

def correct_grammar(text):
    full_corrected_text = ""
    chunk_size = 200
    for start in range(0, len(text), chunk_size):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        result = grammar_tt.generate_text(f"grammar: {chunk}", args=grammar_args)
        corrected_chunk = result.text
        full_corrected_text += corrected_chunk
    return full_corrected_text

def score_essay(text):
    encoded_input = essay_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    device = 'cpu'
    essay_model.to(device)
    with torch.no_grad():
        outputs = essay_model(**encoded_input)
    predictions = outputs.logits.squeeze()
    predicted_scores = predictions.numpy()
    item_names = ["Cohesion", "Syntax", "Vocabulary", "Phraseology", "Grammar", "Conventions"]
    scaled_scores = 2.25 * predicted_scores - 1.25
    rounded_scores = [round(score * 2) / 2 for score in scaled_scores]

    return dict(zip(item_names, rounded_scores))

st.title("Text Resources Apps")

col1, col2, col3 = st.columns(3)

with col1:
    st.header("News Classifier")
    st.write("Upload a PDF or enter text to classify it as real or fake news.")
    if 'pdf_file' not in st.session_state:
        st.session_state.pdf_file = None
    uploaded_file = st.file_uploader("Upload a PDF (only one at a time)", type="pdf", key="file_uploader")
    if uploaded_file:
        st.session_state.pdf_file = uploaded_file
    if 'title_input' not in st.session_state:
        st.session_state.title_input = ""
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    title_input = st.text_input("Enter the title of the news ", value=st.session_state.title_input, key="title_input")
    text_input = st.text_area("Enter the body of the news ", value=st.session_state.text_input, key="text_input")
    if st.button("Submit", key="submit_news"):
        try:
            if st.session_state.pdf_file:
                pdf_reader = PdfReader(st.session_state.pdf_file)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                title_input = "Uploaded News"
                text_input = pdf_text
    
            if text_input:
                prediction = predict_fake(title_input or "No Title", text_input)
                st.write("Prediction: ", "Real" if prediction["Real"] > prediction["Fake"] else "Fake")
                st.write("Probabilities:", prediction)
            else:
                st.write("Please provide text either by uploading a PDF or entering it directly.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col2:
    st.header("Grammar Checker")
    st.write("Upload a PDF or enter text to check for grammatical errors.")
    
    if 'grammar_input' not in st.session_state:
        st.session_state.grammar_input = ""
    if 'grammar_pdf' not in st.session_state:
        st.session_state.grammar_pdf = None
    
    uploaded_grammar_file = st.file_uploader("Upload a PDF (only one at a time)", type="pdf", key="file_uploader_grammar")
    if uploaded_grammar_file:
        st.session_state.grammar_pdf = uploaded_grammar_file
    grammar_input = st.text_area("Enter the text to check ", value=st.session_state.grammar_input, key="grammar_input")
    
    if st.button("Check Grammar", key="check_grammar"):
        try:
            if st.session_state.grammar_pdf:
                pdf_reader = PdfReader(st.session_state.grammar_pdf)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                grammar_input = pdf_text
            
            if grammar_input:
                corrected_text = correct_grammar(grammar_input)
                st.write("Corrected Text:")
                st.text_area("Corrected Text", corrected_text, height=400)
            else:
                st.write("Please provide text to check.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

with col3:
    st.header("Essay Scorer and Grader")
    st.write("Upload a PDF or enter text to score your essay.")
    
    if 'essay_input' not in st.session_state:
        st.session_state.essay_input = ""
    if 'essay_pdf' not in st.session_state:
        st.session_state.essay_pdf = None
    
    uploaded_essay_file = st.file_uploader("Upload a PDF (only one at a time)", type="pdf", key="file_uploader_essay")
    if uploaded_essay_file:
        st.session_state.essay_pdf = uploaded_essay_file
    essay_input = st.text_area("Enter the essay text ", value=st.session_state.essay_input, key="essay_input")
    
    if st.button("Score Essay", key="score_essay"):
        try:
            if st.session_state.essay_pdf:
                pdf_reader = PdfReader(st.session_state.essay_pdf)
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                essay_input = pdf_text
            
            if essay_input:
                scores = score_essay(essay_input)
                st.write("Essay Scores:")
                for item, score in scores.items():
                    st.write(f"{item}: {score:.1f}")
            else:
                st.write("Please provide an essay to score.")
        except Exception as e:
            st.error(f"An error occurred: {e}")