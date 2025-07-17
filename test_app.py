import streamlit as st
import os
import io
from pathlib import Path


# Simple test version without AI dependencies
def main():
    """Simple test version of the app"""
    st.set_page_config(
        page_title="AI Resume Job Matcher - Test",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""

    st.title("🤖 AI Resume Job Matcher - Test Version")
    st.markdown("---")

    # Simple navigation
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📄 Upload Test", "🔍 Text Display"])

    with tab1:
        st.header("Welcome! 🎯")
        st.markdown("""
        ### This is a test version to verify:
        - ✅ Streamlit is working
        - ✅ File uploads work
        - ✅ Basic text processing works

        ### Next Steps:
        1. Test file upload in the "Upload Test" tab
        2. If this works, we'll add the AI features
        """)

        st.success("✅ Streamlit is working perfectly!")

    with tab2:
        st.header("📄 File Upload Test")

        # Test file uploader
        uploaded_file = st.file_uploader(
            "Upload any text file for testing",
            type=['txt', 'pdf', 'docx'],
            help="This is just testing file upload functionality"
        )

        if uploaded_file is not None:
            st.success(f"✅ File uploaded successfully: {uploaded_file.name}")

            # Try to read basic file info
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # Reset file pointer

            st.info(f"File size: {file_size} bytes")

            # Test text extraction for .txt files only
            if uploaded_file.name.endswith('.txt'):
                try:
                    content = uploaded_file.read().decode('utf-8')
                    st.session_state.resume_text = content
                    st.success("✅ Text extracted successfully!")
                except Exception as e:
                    st.error(f"Error reading text: {e}")
            else:
                st.info("📋 For full PDF/Word support, we need the AI modules.")

    with tab3:
        st.header("🔍 Text Display Test")

        if st.session_state.resume_text:
            st.success("✅ Text available in session state!")

            # Show basic stats
            word_count = len(st.session_state.resume_text.split())
            char_count = len(st.session_state.resume_text)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", word_count)
            with col2:
                st.metric("Character Count", char_count)

            # Show text content
            st.markdown("### 📝 Content Preview:")
            st.text_area("Extracted Text", st.session_state.resume_text, height=200)

        else:
            st.warning("No text available. Upload a .txt file in the Upload Test tab.")

    # Test results
    st.markdown("---")
    st.markdown("### 🧪 System Test Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("✅ Streamlit: Working")

    with col2:
        upload_status = "✅ Working" if 'uploaded_file' in locals() else "⏳ Not tested"
        st.info(f"📁 File Upload: {upload_status}")

    with col3:
        text_status = "✅ Working" if st.session_state.resume_text else "⏳ No text"
        st.info(f"📝 Text Processing: {text_status}")


if __name__ == "__main__":
    main()