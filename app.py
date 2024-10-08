import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords', quiet=True)

# Initialize stemmer and stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def read_excel(file, sheet_name=None):
    """Read an Excel file and return its contents as a DataFrame."""
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
        return df
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
    """Read multiple Excel files and combine their text data from all sheets, disregarding empty sheets."""
    combined_texts = []
    raw_texts = []
    filenames = []

    for file in uploaded_files:
        # Get the sheet names from the Excel file
        all_sheets = pd.ExcelFile(file).sheet_names
        st.write(f"Processing file: {file.name} with sheets: {all_sheets}")  # Debug message
        
        for sheet in all_sheets:
            df = read_excel(file, sheet_name=sheet)
            # Drop empty rows and columns
            df.dropna(how='all', inplace=True)  # Drop rows where all elements are NaN
            df.dropna(axis=1, how='all', inplace=True)  # Drop columns where all elements are NaN

            if df.empty:
                st.warning(f"{sheet} is empty after removing empty rows and columns in file '{file.name}'.")
                continue  # Skip empty sheets

            # Filter out NaN values and convert to strings
            df = df.fillna('')  # Replace NaN with empty strings
            combined_text = ' '.join(df.astype(str).values.flatten())  # Flatten the DataFrame into a single string
            combined_texts.append(preprocess_text(combined_text))
            raw_texts.append(combined_text)  # Keep raw text for display
            filenames.append(file.name)  # Store the filename for reference

    return combined_texts, raw_texts, filenames

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
    """Extract all occurrences of the keyword or phrase from the raw texts."""
    snippets_dict = {}
    for text, filename in zip(raw_texts, filenames):
        occurrences = [m.start() for m in re.finditer(re.escape(keyword.lower()), text.lower())]
        if occurrences:  # Only proceed if the keyword is found
            snippets_dict[filename] = snippets_dict.get(filename, set())  # Use set to avoid duplicates
            for start in occurrences:
                start_index = max(start - 30, 0)
                end_index = min(start + len(keyword) + 30, len(text))
                snippet = text[start_index:end_index]
                highlighted_snippet = snippet.replace(keyword, f"<span style='color: red; font-weight: bold;'>{keyword}</span>")
                snippets_dict[filename].add(highlighted_snippet)  # Add each occurrence

    # Convert sets back to lists for rendering
    for filename in snippets_dict:
        snippets_dict[filename] = list(snippets_dict[filename])
    
    return snippets_dict

# Streamlit app
st.title("Excel Keyword Search App")
st.write("Upload multiple Excel files and enter keywords or phrases to search.")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

# Initialize session state variables
if 'tfidf_matrix' not in st.session_state:
    st.session_state.tfidf_matrix = None
    st.session_state.vectorizer = None
    st.session_state.combined_texts = []
    st.session_state.raw_texts = []
    st.session_state.filenames = []

if uploaded_files:
    # Process new files
    combined_texts, raw_texts, filenames = process_files(uploaded_files)
    st.session_state.combined_texts.extend(combined_texts)
    st.session_state.raw_texts.extend(raw_texts)
    st.session_state.filenames.extend(filenames)

    if st.session_state.combined_texts:
        # Create TF-IDF matrix and vectorizer only if there are texts
        if st.session_state.tfidf_matrix is None:  # Calculate only if not already done
            st.session_state.tfidf_matrix, st.session_state.vectorizer = create_tfidf_matrix(st.session_state.combined_texts)

    keyword = st.text_input("Enter keyword or phrase to search:", placeholder="e.g., 'data analysis'")

    if st.button("Search"):
        if st.session_state.tfidf_matrix is not None:
            with st.spinner('Searching...'):
                results = search_keyword(keyword, st.session_state.tfidf_matrix, st.session_state.vectorizer)

                # Display results
                st.write("Search Results:")
                filtered_results = []

                # Loop through the results to create filtered results based on scores
                for idx, score in enumerate(results):
                    if score > 0:  # If the score is greater than zero, add to filtered results
                        filtered_results.append((st.session_state.filenames[idx], score))

                # Ensure there are valid results before extracting snippets
                if filtered_results:
                    snippets_dict = extract_relevant_snippets(st.session_state.raw_texts, keyword)

                    # Display snippets for each file with results
                    for filename, score in filtered_results:
                        if filename in snippets_dict:
                            snippets = snippets_dict[filename]
                            snippet_html = "<br>".join(snippets)  # Join snippets with line breaks
                            st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                                         f"<strong>File:</strong> {filename} | <strong>Relevance Score:</strong> {score:.4f}<br>"
                                         f"<strong>Snippets:</strong><br>{snippet_html}</div>", unsafe_allow_html=True)

                    # Provide a download option for results
                    result_df = pd.DataFrame(filtered_results, columns=["File", "Relevance Score"])
                    st.download_button("Download Results", result_df.to_csv(index=False).encode('utf-8'), "search_results.csv", "text/csv")
                else:
                    st.write("No results found.")
        else:
            st.warning("No text data to search. Please upload files.")
