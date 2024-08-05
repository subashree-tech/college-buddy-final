import streamlit as st
import pandas as pd
import sqlite3

def get_database_connection():
    conn = sqlite3.connect('college_buddy.db', check_same_thread=False)
    return conn

def get_all_documents():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, tags, links FROM documents WHERE tags != '' AND links != ''")
    return c.fetchall()

def insert_document(title, tags, links):
    if tags.strip() and links.strip():
        conn = get_database_connection()
        c = conn.cursor()
        c.execute("INSERT INTO documents (title, tags, links) VALUES (?, ?, ?)",
                  (title, tags, links))
        conn.commit()
        return True
    return False

def delete_document_by_title(title):
    conn = get_database_connection()
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE title = ?", (title,))
    if c.rowcount > 0:
        conn.commit()
        return True
    return False

def show():
    st.title("Database Management")

    # Form for adding new documents
    st.header("Add New Document")
    new_doc_title = st.text_input("Document Title", key="add_title")
    new_doc_tags = st.text_input("Tags (comma-separated)")
    new_doc_links = st.text_input("Links")

    if st.button("Add Document"):
        if new_doc_title and new_doc_tags and new_doc_links:
            if insert_document(new_doc_title, new_doc_tags, new_doc_links):
                st.success(f"Document '{new_doc_title}' added successfully!")
                st.experimental_rerun()
            else:
                st.warning("Failed to add document. Please check your inputs.")
        else:
            st.warning("Please fill in all fields.")

    # Form for deleting documents
    st.header("Delete Document")
    del_doc_title = st.text_input("Enter the title of the document to delete", key="del_title")

    if st.button("Delete Document"):
        if del_doc_title:
            if delete_document_by_title(del_doc_title):
                st.success(f"Document '{del_doc_title}' deleted successfully!")
                st.experimental_rerun()
            else:
                st.warning(f"No document found with the title '{del_doc_title}'.")
        else:
            st.warning("Please enter a document title to delete.")

    # Display existing documents
    st.header("Existing Documents")
    documents = get_all_documents()
    if documents:
        df = pd.DataFrame(documents, columns=['ID', 'Title', 'Tags', 'Links'])
        st.table(df)
    else:
        st.write("The database is empty.")

if __name__ == "__main__":
    show()
