import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from openpyxl import load_workbook
from io import BytesIO

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords')

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

def extract_images(file, sheet_name=None):
    """Extract images from the specified sheet in an Excel file."""
    images = []
    try:
        wb = load_workbook(file)
        sheet = wb[sheet_name]
        
        for img in sheet._images:
            # Convert image to BytesIO to display with Streamlit
            image_stream = BytesIO()
            img.image.save(image_stream, format='PNG')  # Save the image to the stream in PNG format
            image_stream.seek(0)  # Move to the beginning of the stream
            images.append(image_stream)
            
    except Exception as e:
        st.error(f"Error extracting images from {file.name}: {e}")

    return images

def preprocess_text(text):
    """Preprocess the text data."""
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join(stemmer.stem(word) for word in text.split() if word not in stop_words)  # Remove stop words and stem
    return text

def process_files(uploaded_files):
    """Read multiple Excel files and combine their text data from all sheets."""
    combined_texts = []
    raw_texts = []
    all_images = []

    for file in uploaded_files:
        # Get the sheet names from the Excel file
        all_sheets = pd.ExcelFile(file).sheet_names
        for sheet in all_sheets:
            df = read_excel(file, sheet_name=sheet)
            if not df.empty:
                # Drop empty rows and columns
                df.dropna(how='all', inplace=True)  # Drop rows where all elements are NaN
                df.dropna(axis=1, how='all', inplace=True)  # Drop columns where all elements are NaN
                
                if df.empty:
                    st.warning(f"{sheet} is empty after removing empty rows and columns in file '{file.name}'.")
                    continue

                # Filter out NaN values and convert to strings
                df = df.fillna('')  # Replace NaN with empty strings
                combined_text = ' '.join(df.astype(str).values.flatten())  # Flatten the DataFrame into a single string
                combined_texts.append(preprocess_text(combined_text))
                raw_texts.append(combined_text)  # Keep raw text for display
                
                # Extract images from the current sheet
                images = extract_images(file, sheet_name=sheet)
                all_images.append(images)
            else:
                st.warning(f"{sheet} is empty in file '{file.name}'.")

    return combined_texts, raw_texts, all_images

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
    snippets = []
    for text in raw_texts:
        occurrences = [m.start() for m in re.finditer(re.escape(keyword.lower()), text.lower())]
        if occurrences:  # Only proceed if the keyword is found
            for start in occurrences:
                start_index = max(start - 30, 0)
                end_index = min(start + len(keyword) + 30, len(text))
                snippet = text[start_index:end_index]
                highlighted_snippet = snippet.replace(keyword, f"<span style='color: red; font-weight: bold;'>{keyword}</span>")
                snippets.append(highlighted_snippet)  # Append each occurrence
    return snippets

# Streamlit app
st.title("Excel Keyword Search App")
st.write("Upload multiple Excel files and enter keywords or phrases to search.")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    keyword = st.text_input("Enter keyword or phrase to search:", placeholder="e.g., 'data analysis'")

    if st.button("Search"):
        with st.spinner('Processing files...'):
            combined_texts, raw_texts, all_images = process_files(uploaded_files)
            if not combined_texts:  # Check if there are any processed texts
                st.error("No valid text found in the uploaded files.")
            else:
                tfidf_matrix, vectorizer = create_tfidf_matrix(combined_texts)
                results = search_keyword(keyword, tfidf_matrix, vectorizer)

                # Display results
                st.write("Search Results:")
                filtered_results = [(uploaded_files[idx].name, score) for idx, score in enumerate(results) if score > 0]

                if filtered_results:
                    for (filename, score), snippets, images in zip(filtered_results, extract_relevant_snippets(raw_texts, keyword), all_images):
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                                     f"<strong>File:</strong> {filename} | <strong>Relevance Score:</strong> {score:.4f}<br>"
                                     f"<strong>Snippets:</strong></div>", unsafe_allow_html=True)
                        for snippet in snippets:
                            st.markdown(f"<div style='padding-left: 10px;'>{snippet}</div>", unsafe_allow_html=True)

                        # Display images if available
                        for image in images:
                            st.image(image, caption='Image from Excel', use_column_width=True)

                    # Provide a download option for results
                    result_df = pd.DataFrame(filtered_results, columns=["File", "Relevance Score"])
                    st.download_button("Download Results", result_df.to_csv(index=False).encode('utf-8'), "search_results.csv", "text/csv")
                else:
                    st.write("No results found.")
