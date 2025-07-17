import streamlit as st
import os
import io
import pandas as pd
from utils.text_extractor import TextExtractor
from models.ai_models import SimpleAIModel


# Initialize session state FIRST - before any other operations
def initialize_session_state():
    """Initialize all session state variables"""
    if 'text_extractor' not in st.session_state:
        st.session_state.text_extractor = None

    if 'ai_analyzer' not in st.session_state:
        st.session_state.ai_analyzer = None

    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = ""

    if 'resume_sections' not in st.session_state:
        st.session_state.resume_sections = {}

    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False


@st.cache_resource
def load_ai_models():
    """Load AI models with caching and error handling"""
    try:
        text_extractor = TextExtractor()
        ai_analyzer = SimpleAIModel()
        return text_extractor, ai_analyzer, True
    except Exception as e:
        st.error(f"Error loading AI models: {str(e)}")
        return None, None, False


def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="AI Resume Job Matcher",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state FIRST
    initialize_session_state()

    # Main header
    st.title("🤖 AI Resume Job Matcher")
    st.markdown("---")

    # Load models if not already loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            text_extractor, ai_analyzer, success = load_ai_models()

            if success:
                st.session_state.text_extractor = text_extractor
                st.session_state.ai_analyzer = ai_analyzer
                st.session_state.models_loaded = True
                st.success("✅ AI models loaded successfully!")
            else:
                st.error("❌ Failed to load AI models. Some features may not work.")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📄 Upload Resume", "🔍 Job Analysis", "📊 Match Results", "⚙️ Settings"]
    )

    # Route to different pages
    if page == "🏠 Home":
        show_home_page()
    elif page == "📄 Upload Resume":
        show_upload_page()
    elif page == "🔍 Job Analysis":
        show_job_analysis_page()
    elif page == "📊 Match Results":
        show_results_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_home_page():
    """Display the home page"""
    st.header("Welcome to AI Resume Job Matcher! 🎯")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### What This App Does:

        🔍 **Smart Resume Analysis**: Upload your resume and get AI-powered insights

        🎯 **Job Matching**: Compare your resume against job descriptions using advanced AI

        📊 **Detailed Reports**: Get match percentages and improvement recommendations

        🚀 **Career Enhancement**: Identify skill gaps and optimization opportunities

        ### How It Works:

        1. **Upload** your resume (PDF, Word, or Text)
        2. **Analyze** your resume with AI to extract key information
        3. **Compare** against job descriptions using semantic similarity
        4. **Receive** detailed match scores and recommendations

        ### Features:
        - Support for PDF, Word, and text files
        - Advanced AI text analysis using keyword matching
        - Job matching with detailed analysis
        - Section-by-section analysis
        - Personalized improvement recommendations
        """)

    with col2:
        st.markdown("### 📊 Quick Stats")

        # Check if models are loaded
        models_loaded = st.session_state.get('models_loaded', False)

        if models_loaded:
            st.success("✅ AI Models Loaded")
        else:
            st.warning("⏳ Loading AI Models...")

        resume_uploaded = bool(st.session_state.get('resume_text', ''))
        st.info(f"📁 Resume Uploaded: {'✅' if resume_uploaded else '❌'}")

        st.markdown("### 🚀 Get Started")
        if st.button("Upload Your Resume", type="primary"):
            st.info("👆 Use the sidebar to navigate to 'Upload Resume'")


def show_upload_page():
    """Display the resume upload page"""
    st.header("📄 Upload Your Resume")

    if not st.session_state.get('models_loaded', False):
        st.warning("AI models are still loading. Please wait or refresh the page.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose your resume file",
        type=['pdf', 'docx', 'doc', 'txt'],
        help="Supported formats: PDF, Word documents, and text files"
    )

    if uploaded_file is not None:
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")

        # Extract text
        with st.spinner("Extracting text from your resume..."):
            try:
                # Get file extension
                file_extension = os.path.splitext(uploaded_file.name)[1]

                # Create BytesIO object
                file_bytes = io.BytesIO(uploaded_file.read())

                # Extract text
                text_extractor = st.session_state.text_extractor
                if text_extractor:
                    resume_text = text_extractor.extract_text(file_bytes, file_extension)

                    if resume_text and not resume_text.startswith(
                            "PDF support not available") and not resume_text.startswith(
                            "Word document support not available") and not resume_text.startswith("Error"):
                        st.session_state.resume_text = resume_text

                        # Extract sections
                        st.session_state.resume_sections = text_extractor.extract_resume_sections(resume_text)

                        st.success("✅ Resume processed successfully!")
                    else:
                        st.error(
                            "❌ Could not extract text from the file. Please check if required libraries are installed.")
                        if file_extension == '.pdf':
                            st.info("For PDF support, install: `pip install PyPDF2`")
                        elif file_extension in ['.docx', '.doc']:
                            st.info("For Word document support, install: `pip install python-docx`")
                        return
                else:
                    st.error("Text extractor not available. Please refresh the page.")
                    return

            except Exception as e:
                st.error(f"Error processing resume: {str(e)}")
                return

    # Display extracted content if available
    resume_text = st.session_state.get('resume_text', '')
    if resume_text:
        st.markdown("### 📋 Resume Analysis")

        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            word_count = len(resume_text.split())
            st.metric("Word Count", word_count)

        with col2:
            sections = st.session_state.get('resume_sections', {})
            sections_found = len([s for s in sections.values() if s.strip()])
            st.metric("Sections Found", sections_found)

        with col3:
            text_extractor = st.session_state.get('text_extractor')
            if text_extractor:
                keywords = text_extractor.extract_keywords(resume_text)
                st.metric("Key Skills", len(keywords))
            else:
                st.metric("Key Skills", "N/A")

        # Show extracted sections
        st.markdown("### 📑 Extracted Sections")

        sections = st.session_state.get('resume_sections', {})
        for section_name, section_content in sections.items():
            if section_content.strip():
                with st.expander(f"{section_name.title()} Section"):
                    display_content = section_content[:500] + "..." if len(section_content) > 500 else section_content
                    st.write(display_content)

        # Show keywords
        text_extractor = st.session_state.get('text_extractor')
        if text_extractor:
            keywords = text_extractor.extract_keywords(resume_text)
            if keywords:
                st.markdown("### 🔑 Detected Keywords")
                st.write(", ".join(keywords[:15]))

        # Next steps
        st.markdown("### ✅ Next Steps")
        st.info("Your resume is ready! Go to 'Job Analysis' to compare it with job descriptions.")


def show_job_analysis_page():
    """Display the job analysis page"""
    st.header("🔍 Job Analysis")

    resume_text = st.session_state.get('resume_text', '')
    if not resume_text:
        st.warning("Please upload your resume first!")
        st.info("👆 Use the sidebar to navigate to 'Upload Resume'")
        return

    st.markdown("### Compare Your Resume with Job Descriptions")

    # Job input methods
    input_method = st.radio(
        "How would you like to add job information?",
        ["✍️ Paste Job Description", "📂 Upload Job File"]
    )

    job_title = ""
    job_description = ""
    job_requirements = ""

    if input_method == "✍️ Paste Job Description":
        col1, col2 = st.columns(2)

        with col1:
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste the full job description here..."
            )

        with col2:
            job_requirements = st.text_area(
                "Requirements (Optional)",
                height=200,
                placeholder="Paste specific requirements or leave this section empty"
            )

    elif input_method == "📂 Upload Job File":
        uploaded_job_file = st.file_uploader(
            "Upload Job Description File",
            type=['txt', 'pdf', 'docx'],
            help="Upload a file containing the job description"
        )

        if uploaded_job_file:
            try:
                file_extension = os.path.splitext(uploaded_job_file.name)[1]
                file_bytes = io.BytesIO(uploaded_job_file.read())

                text_extractor = st.session_state.get('text_extractor')
                if text_extractor:
                    job_description = text_extractor.extract_text(file_bytes, file_extension)

                    if job_description and not job_description.startswith(
                            "PDF support not available") and not job_description.startswith(
                            "Word document support not available") and not job_description.startswith("Error"):
                        st.success("Job description extracted successfully!")
                        st.text_area("Extracted Job Description", job_description, height=150)
                    else:
                        st.error("Could not extract text from the file.")
                else:
                    st.error("Text extractor not available.")

            except Exception as e:
                st.error(f"Error extracting job description: {str(e)}")

    # Analysis button
    analysis_disabled = not job_description.strip() or not st.session_state.get('models_loaded', False)

    if st.button("🚀 Analyze Job Match", type="primary", disabled=analysis_disabled):
        if not job_description.strip():
            st.error("Please provide a job description!")
            return

        if not st.session_state.get('models_loaded', False):
            st.error("AI models are not loaded yet. Please wait or refresh the page.")
            return

        with st.spinner("Analyzing job match with AI... This may take a moment."):
            try:
                # Perform analysis
                ai_analyzer = st.session_state.get('ai_analyzer')
                resume_sections = st.session_state.get('resume_sections', {})

                if ai_analyzer:
                    analysis_result = ai_analyzer.analyze_resume_job_match(
                        resume_text=resume_text,
                        resume_sections=resume_sections,
                        job_description=job_description,
                        job_title=job_title,
                        job_requirements=job_requirements
                    )

                    # Store results
                    st.session_state.current_analysis = {
                        'job_title': job_title,
                        'job_description': job_description,
                        'job_requirements': job_requirements,
                        'analysis': analysis_result
                    }

                    st.success("✅ Analysis complete! Check the Match Results page.")
                else:
                    st.error("AI analyzer not available.")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Try reloading the page or check your inputs.")


def show_results_page():
    """Display analysis results"""
    st.header("📊 Match Results")

    current_analysis = st.session_state.get('current_analysis')
    if not current_analysis:
        st.warning("No analysis results available. Please analyze a job first!")
        st.info("👆 Use the sidebar to navigate to 'Job Analysis'")
        return

    analysis_data = current_analysis
    analysis = analysis_data['analysis']

    # Job info
    st.markdown("### 💼 Job Information")
    if analysis_data.get('job_title'):
        st.markdown(f"**Job Title:** {analysis_data['job_title']}")

    # Overall match score
    st.markdown("### 🎯 Overall Match Score")

    match_percentage = analysis['match_percentage']
    match_level = analysis.get('match_level', 'Unknown')

    # Color-coded progress bar
    if match_percentage >= 70:
        color = "green"
        status = "Strong Match! 🔥"
    elif match_percentage >= 50:
        color = "orange"
        status = "Good Match 👍"
    elif match_percentage >= 30:
        color = "yellow"
        status = "Fair Match 🤔"
    else:
        color = "red"
        status = "Weak Match 📈"

    # Display match percentage
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.metric("Match Score", f"{match_percentage}%", help=status)
        st.progress(match_percentage / 100)
        st.markdown(f"**Status:** {status}")
        st.markdown(f"**Match Level:** {match_level}")

    # Skills analysis
    st.markdown("### 🛠️ Skills Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Common Skills")
        common_skills = analysis.get('common_skills', [])
        if common_skills:
            for skill in common_skills:
                st.markdown(f"• {skill}")
        else:
            st.info("No common skills detected")

    with col2:
        st.markdown("#### ❌ Missing Skills")
        missing_skills = analysis.get('missing_skills', [])
        if missing_skills:
            for skill in missing_skills[:10]:  # Show top 10
                st.markdown(f"• {skill}")
        else:
            st.info("No missing skills detected")

    # Section analysis
    st.markdown("### 📑 Section-by-Section Analysis")

    section_matches = analysis.get('section_matches', {})
    if section_matches:
        for section, score in section_matches.items():
            if score > 0:  # Only show sections with content
                percentage = int(score * 100)
                st.markdown(f"**{section.title()}:** {percentage}%")
                st.progress(score)
    else:
        st.info("No section analysis available")

    # Keywords analysis
    st.markdown("### 🔑 Keywords Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Common Keywords")
        common_keywords = analysis.get('common_keywords', {})
        if common_keywords:
            # Show top 10 common keywords
            for keyword, data in list(common_keywords.items())[:10]:
                st.markdown(f"• **{keyword}** (appears {data['resume_freq']} times)")
        else:
            st.info("No common keywords found")

    with col2:
        st.markdown("#### Missing Important Keywords")
        missing_keywords = analysis.get('missing_keywords', {})
        if missing_keywords:
            # Show top 10 missing keywords
            for keyword, freq in list(missing_keywords.items())[:10]:
                st.markdown(f"• **{keyword}** (mentioned {freq} times in job)")
        else:
            st.info("No missing keywords detected")

    # Recommendations
    st.markdown("### 💡 Recommendations")

    recommendations = analysis.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific recommendations available")

    # Export results
    st.markdown("### 📄 Export Results")

    if st.button("📊 Generate Detailed Report", type="secondary"):
        # Create a comprehensive report
        report_data = {
            'Job Title': analysis_data.get('job_title', 'N/A'),
            'Match Score': f"{match_percentage}%",
            'Match Level': match_level,
            'Status': status,
            'Common Skills': ', '.join(common_skills) if common_skills else 'None',
            'Missing Skills': ', '.join(missing_skills[:5]) if missing_skills else 'None',
            'Top Recommendations': '; '.join(recommendations[:3]) if recommendations else 'None'
        }

        # Display as dataframe
        df = pd.DataFrame([report_data])
        st.dataframe(df, use_container_width=True)

        # Download button would go here in a real implementation
        st.info("💡 Tip: Take a screenshot of these results for your records!")


def show_settings_page():
    """Display settings page"""
    st.header("⚙️ Settings")

    st.markdown("### Application Settings")

    # Model status
    st.markdown("#### 🤖 AI Model Status")
    models_loaded = st.session_state.get('models_loaded', False)

    if models_loaded:
        st.success("✅ AI models are loaded and ready")
        if st.button("🔄 Reload Models"):
            st.session_state.models_loaded = False
            st.rerun()
    else:
        st.warning("⚠️ AI models are not loaded")
        if st.button("🚀 Load Models"):
            with st.spinner("Loading models..."):
                text_extractor, ai_analyzer, success = load_ai_models()
                if success:
                    st.session_state.text_extractor = text_extractor
                    st.session_state.ai_analyzer = ai_analyzer
                    st.session_state.models_loaded = True
                    st.success("Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("Failed to load models")

    # Data management
    st.markdown("#### 🗂️ Data Management")

    resume_uploaded = bool(st.session_state.get('resume_text', ''))
    analysis_available = st.session_state.get('current_analysis') is not None

    st.info(f"📄 Resume Status: {'Uploaded' if resume_uploaded else 'Not uploaded'}")
    st.info(f"📊 Analysis Status: {'Available' if analysis_available else 'No analysis'}")

    if st.button("🗑️ Clear All Data", type="secondary"):
        # Clear all session state data
        for key in ['resume_text', 'resume_sections', 'current_analysis']:
            if key in st.session_state:
                del st.session_state[key]
        st.success("All data cleared!")
        st.rerun()

    # App information
    st.markdown("#### ℹ️ About")
    st.markdown("""
    **AI Resume Job Matcher v1.0**

    This application helps you analyze how well your resume matches job descriptions using AI-powered text analysis.

    **Features:**
    - Resume text extraction from PDF, Word, and text files
    - AI-powered job matching analysis  
    - Skills gap analysis
    - Personalized recommendations
    - Section-by-section breakdown

    **Technical Stack:**
    - Streamlit for the web interface
    - PyPDF2 for PDF processing
    - python-docx for Word document processing
    - Custom AI models for text analysis
    """)

    # Dependencies check
    st.markdown("#### 🔧 Dependencies Check")

    dependencies = {
        'PyPDF2': False,
        'python-docx': False,
        'pandas': True,  # Always available in streamlit
        'streamlit': True  # Obviously available
    }

    try:
        import PyPDF2
        dependencies['PyPDF2'] = True
    except ImportError:
        pass

    try:
        import docx
        dependencies['python-docx'] = True
    except ImportError:
        pass

    for dep, available in dependencies.items():
        if available:
            st.success(f"✅ {dep}")
        else:
            st.error(f"❌ {dep} - Install with: pip install {dep}")


if __name__ == "__main__":
    main()