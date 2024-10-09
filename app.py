import pandas as pd
import streamlit as st
import sqlite3
import re
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import text

# Set up the SQLAlchemy ORM model
Base = declarative_base()

class Sheet(Base):
    __tablename__ = 'sheets'
    id = Column(Integer, primary_key=True)
    file_name = Column(String, nullable=False)
    sheet_name = Column(String, nullable=False)
    snippets = relationship("Snippet", back_populates="sheet")

class Snippet(Base):
    __tablename__ = 'snippets'
    id = Column(Integer, primary_key=True)
    content = Column(String, nullable=False)
    sheet_id = Column(Integer, ForeignKey('sheets.id'))
    sheet = relationship("Sheet", back_populates="snippets")

# Set up the SQLite database and SQLAlchemy ORM
db_engine = create_engine('sqlite:///search_app.db')
Base.metadata.create_all(db_engine)
Session = sessionmaker(bind=db_engine)
session = Session()

def read_excel_to_orm(file, sheet_name=None):
    """Read an Excel file and save its content in the ORM."""
    try:
        df = pd.read_excel(file, sheet_name=sheet_name)
        df.dropna(how='all', inplace=True)  # Remove empty rows
        df.dropna(axis=1, how='all', inplace=True)  # Remove empty columns

        if df.empty:
            st.warning(f"{sheet_name or file.name} is empty.")
            return

        # Combine all text into one column
        df['combined_text'] = df.astype(str).agg(' '.join, axis=1)

        # Create ORM objects
        sheet = Sheet(file_name=file.name, sheet_name=sheet_name)
        session.add(sheet)
        session.commit()

        # Insert combined text snippets into the ORM
        for text_row in df['combined_text']:
            snippet = Snippet(content=text_row, sheet=sheet)
            session.add(snippet)

        session.commit()

    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")

def load_files_to_db(uploaded_files):
    """Load all uploaded Excel files into the ORM."""
    for file in uploaded_files:
        try:
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names
            st.write(f"Processing file: {file.name} with sheets: {sheet_names}")

            for sheet in sheet_names:
                read_excel_to_orm(file, sheet_name=sheet)

        except Exception as e:
            st.error(f"Error loading {file.name}: {e}")

def search_keyword_advanced(keyword, proximity=None):
    """Search for a keyword using advanced FTS features like proximity search."""
    results = []
    try:
        query_str = f"SELECT content FROM snippets WHERE content MATCH :keyword"
        if proximity:
            keyword = f'"{keyword}" NEAR/{proximity}'

        query = text(query_str)
        snippets = session.execute(query, {'keyword': keyword}).fetchall()

        results = [snippet[0] for snippet in snippets]
    except Exception as e:
        st.error(f"Error searching in the database: {e}")

    return results

def highlight_keyword(text, keyword):
    """Highlight occurrences of the keyword in the text."""
    highlighted = re.sub(f"({re.escape(keyword)})", r"<span style='color: red;'>\1</span>", text, flags=re.IGNORECASE)
    return highlighted

def download_results(results, file_name='search_results.csv'):
    """Provide a download link for search results as a CSV file."""
    df = pd.DataFrame(results, columns=["Snippet"])
    return df.to_csv(index=False).encode('utf-8')

# Streamlit app
st.title("Excel Keyword Search with SQL Full-Text Search and Advanced Queries")

uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    load_files_to_db(uploaded_files)

    keyword = st.text_input("Enter keyword or phrase to search:", placeholder="e.g., 'data analysis'")
    proximity = st.slider("Proximity search (words apart)", min_value=1, max_value=10, value=5, step=1)

    if st.button("Search"):
        if keyword:
            st.write(f"Searching for '{keyword}' in the database with proximity {proximity}...")
            with st.spinner("Searching..."):
                results = search_keyword_advanced(keyword, proximity=proximity)
                if results:
                    for snippet in results:
                        highlighted_snippet = highlight_keyword(snippet, keyword)
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0;'>"
                                    f"<strong>Snippet:</strong> {highlighted_snippet}</div>", unsafe_allow_html=True)

                    # Provide a download option for results
                    st.download_button("Download Results", download_results(results), "search_results.csv", "text/csv")
                else:
                    st.write("No results found.")
        else:
            st.warning("Please enter a keyword to search.")
