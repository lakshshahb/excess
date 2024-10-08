import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords')

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def read_excel(file):
    """Read an Excel file and return its contents as a DataFrame."""
    if not file.name.endswith('.xlsx'):
        st.error("Uploaded file is not an Excel file. Please upload a .xlsx file.")
        return pd.DataFrame()
    try:
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return pd.DataFrame()

def preprocess_text(text):
    """Preprocess the text data."""
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)  # Remove stop words and stem
    return text

def process_files(uploaded_files):
    """Read multiple Excel files and combine their text data."""
    combined_texts = []
    raw_texts = []
    for file in uploaded_files:
        df = read_excel(file)
        if not df.empty:
            combined_text = ' '.join(df.astype(str).values.flatten())
            combined_texts.append(preprocess_text(combined_text))
            raw_texts.append(combined_text)  # Keep raw text for display
    return combined_texts, raw_texts

def create_tfidf_matrix(texts):
    """Create a TF-IDF matrix from the combined text data with n-grams."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Allow unigrams and bigrams
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def search_keyword(keyword, tfidf_matrix, vectorizer):
    """Search for a keyword or phrase in the TF-IDF matrix and return relevant indices."""
    query_vec = vectorizer.transform([keyword])
    results = (tfidf_matrix * query_vec.T).toarray()
    return results.flatten()

def extract_relevant_snippets(raw_texts, keyword):
    """Extract snippets containing the keyword or phrase from the raw texts."""
    snippets = []
    for text in raw_texts:
        occurrences = [m.start() for m in re.finditer(re.escape(keyword.lower()), text.lower())]
        for start in occurrences:
            start_index = max(start - 30, 0)
            end_index = min(start + len(keyword) + 30, len(text))
            snippet = text[start_index:end_index]
            highlighted_snippet = snippet.replace(keyword, f"<span style='color: red; font-weight: bold;'>{keyword}</span>")
            snippets.append(highlighted_snippet)
    return snippets

# Streamlit app
st.title("Excel Keyword Search App")
st.write("Upload Excel files and enter keywords or phrases to search.")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    keyword = st.text_input("Enter keyword or phrase to search:", placeholder="e.g., 'data analysis'")

    if st.button("Search"):
        with st.spinner('Processing files...'):
            combined_texts, raw_texts = process_files(uploaded_files)
            if not combined_texts:  # Check if there are any processed texts
                st.error("No valid text found in the uploaded files.")
            else:
                tfidf_matrix, vectorizer = create_tfidf_matrix(combined_texts)
                results = search_keyword(keyword, tfidf_matrix, vectorizer)

                # Display results
                st.write("Search Results:")
                filtered_results = [(uploaded_files[idx].name, score) for idx, score in enumerate(results) if score > 0]

                if filtered_results:
                    for (filename, score), snippet in zip(filtered_results, extract_relevant_snippets(raw_texts, keyword)):
                        # Create a box around each result
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                                     f"<strong>File:</strong> {filename} | <strong>Relevance Score:</strong> {score:.4f}<br>"
                                     f"<strong>Snippet:</strong> {snippet}</div>", unsafe_allow_html=True)
                    
                    # Provide a download option for results
                    result_df = pd.DataFrame(filtered_results, columns=["File", "Relevance Score"])
                    st.download_button("Download Results", result_df.to_csv(index=False).encode('utf-8'), "search_results.csv", "text/csv")
                else:
                    st.write("No results found.")

# Run the app using: streamlit run your_script_name.py
