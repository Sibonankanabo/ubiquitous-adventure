import streamlit as st

# Check if the user is logged in
if 'logged_in' in st.session_state and st.session_state['logged_in']:
    st.title("Main Application")
    st.write("Welcome to the main application!")
    st.write(f"You are logged in as {st.session_state['username']}")
    # Add logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.experimental_rerun()
else:
    st.warning("Please log in to access this page.")
    st.stop()
