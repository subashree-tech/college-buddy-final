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
def show():
    st.title("Database Contents")
    documents = get_all_documents()
    if documents:
        df = pd.DataFrame(documents, columns=['ID', 'Title', 'Tags', 'Links'])
        st.dataframe(df)
    else:
        st.write("The database is empty.")
if __name__ == "__main__":
    show()
