import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import re
import nltk

# Ensure you have the NLTK stopwords downloaded
nltk.download('stopwords')

# Initialize SQLAlchemy database setup
DATABASE_URL = 'sqlite:///search_app.db'
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

def create_fts_table():
    """Create a virtual FTS table for snippets."""
    with engine.connect() as connection:
        connection.execute(text("DROP TABLE IF EXISTS snippets;"))  # Clean up existing table
        connection.execute(text("CREATE VIRTUAL TABLE snippets USING fts5(content);"))  # Create FTS table

create_fts_table()

def read_excel(file):
    """Read an Excel file and return its contents as a DataFrame."""
    try:
        df = pd.read_excel(file, sheet_name=None)  # Read all sheets
        return df
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return {}

def preprocess_text(text):
    """Preprocess the text data."""
    text = re.sub(r'\W+', ' ', text)  # Remove non-word characters
    text = text.lower()  # Convert to lowercase
    return text

def process_files(uploaded_files):
    """Read multiple Excel files and combine their text data."""
    combined_texts = []
    
    for file in uploaded_files:
        sheets = read_excel(file)
        for sheet_name, df in sheets.items():
            if not df.empty:
                # Remove empty rows and columns
                df.dropna(how='all', inplace=True)
                df.dropna(axis=1, how='all', inplace=True)

                combined_text = ' '.join(df.astype(str).values.flatten())
                combined_texts.append(preprocess_text(combined_text))

                # Insert processed text into the FTS table
                insert_snippet(combined_text)
            else:
                st.warning(f"{file.name} - Sheet {sheet_name} is empty or could not be read.")
    
    return combined_texts

def insert_snippet(content):
    """Insert a snippet into the FTS table."""
    session = Session()  # Create a session for database operations
    session.execute(text("INSERT INTO snippets (content) VALUES (:content)"), {'content': content})
    session.commit()
    session.close()  # Close the session after the operation

def search_in_db(keyword):
    """Search for a keyword in the FTS table and return results."""
    session = Session()  # Create a new session for the search operation
    query = "SELECT content FROM snippets WHERE content MATCH :keyword"
    results = session.execute(text(query), {'keyword': keyword}).fetchall()
    session.close()  # Close the session after fetching results
    return results

# Streamlit app
st.title("Excel Keyword Search App")
st.write("Upload Excel files and enter keywords or phrases to search.")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    keyword = st.text_input("Enter keyword or phrase to search:", placeholder="e.g., 'data analysis'")

    if st.button("Search"):
        with st.spinner('Processing files...'):
            combined_texts = process_files(uploaded_files)
            if not combined_texts:  # Check if there are any processed texts
                st.error("No valid text found in the uploaded files.")
            else:
                results = search_in_db(keyword)

                # Display results
                st.write("Search Results:")
                if results:
                    for result in results:
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                                     f"<strong>Snippet:</strong> {result[0]}</div>", unsafe_allow_html=True)
                else:
                    st.write("No results found.")
