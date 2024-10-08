# ... (rest of your imports and code)

# Streamlit app
st.title("Excel Keyword Search App")
st.write("Upload multiple Excel files and enter keywords or phrases to search.")

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
                    relevant_snippets = extract_relevant_snippets(raw_texts, keyword)

                    for (filename, score), snippets in zip(filtered_results, relevant_snippets):
                        # Create a single HTML block for the snippets
                        snippet_html = "<br>".join(snippets)  # Join snippets with line breaks
                        st.markdown(f"<div style='border: 1px solid #ddd; padding: 10px; margin: 10px 0; border-radius: 5px;'>"
                                     f"<strong>File:</strong> {filename} | <strong>Relevance Score:</strong> {score:.4f}<br>"
                                     f"<strong>Snippets:</strong><br>{snippet_html}</div>", unsafe_allow_html=True)

                    # Provide a download option for results
                    result_df = pd.DataFrame(filtered_results, columns=["File", "Relevance Score"])
                    st.download_button("Download Results", result_df.to_csv(index=False).encode('utf-8'), "search_results.csv", "text/csv")
                else:
                    st.write("No results found.")
