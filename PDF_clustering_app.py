# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:19:18 2024

@author: 91981
"""

import streamlit as st
import pandas as pd
import pickle
import re
import fitz  

# Load the trained model and vectorizer
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('cluster_names.pkl', 'rb') as names_file:
    cluster_names = pickle.load(names_file)

# Function to extract text from PDFs
def extract_text_from_pdfs(uploaded_files):
    texts = []
    for pdf_file in uploaded_files:
        # Read the PDF file as bytes
        pdf_bytes = pdf_file.read()
        # Open the PDF file from bytes
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            texts.append(text)
    return texts

# Basic text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Streamlit UI
st.title("PDF Clustering App")
st.write("Upload your PDF files to see their clusters and key factors.")

# File uploader for PDF files
uploaded_files = st.file_uploader("Choose PDF files", type='pdf', accept_multiple_files=True)

if uploaded_files:
    # Extract text from the uploaded PDFs
    texts = extract_text_from_pdfs(uploaded_files)
    
    # Preprocess the texts
    cleaned_texts = [preprocess_text(text) for text in texts]
    
    # Remove empty or very short strings
    cleaned_texts = [text for text in cleaned_texts if len(text.strip()) > 2]
    
    # Convert cleaned texts into TF-IDF features
    X = vectorizer.transform(cleaned_texts)
    
    # Predict cluster labels for the uploaded PDFs
    labels = kmeans.predict(X)
    
    # Create a DataFrame to display results
    results_df = pd.DataFrame({
        'PDF File': [file.name for file in uploaded_files],
        'Cluster': labels,
        'Topic': [cluster_names[label] for label in labels]
    })

    # Display the results
    st.subheader("Clustering Results")
    st.write(results_df)

    # Display common keywords for each cluster
    st.subheader("Key Factors for Each Cluster")
    for cluster in set(labels):
        st.write(f"**Cluster {cluster} ({cluster_names[cluster]})**")
        # Get the indices of top terms for the current cluster
        top_term_indices = kmeans.cluster_centers_[cluster].argsort()[-10:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[index] for index in top_term_indices]
        st.write("Common keywords:", top_terms)
