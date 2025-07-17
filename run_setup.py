"""
Test script to verify all dependencies are installed correctly
Run this after installing packages to ensure everything works
"""

import sys
import importlib
from typing import Dict, List, Tuple


def test_imports() -> Dict[str, bool]:
    """Test if all required packages can be imported"""

    # Required packages with their import names
    packages = {
        'streamlit': 'streamlit',
        'sentence-transformers': 'sentence_transformers',
        'chromadb': 'chromadb',
        'transformers': 'transformers',
        'torch': 'torch',
        'PyPDF2': 'PyPDF2',
        'python-docx': 'docx',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'requests': 'requests',
        'pathlib': 'pathlib'
    }

    results = {}

    print("🔍 Testing Package Imports...")
    print("=" * 40)

    for package_name, import_name in packages.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name:<20} - OK")
            results[package_name] = True
        except ImportError as e:
            print(f"❌ {package_name:<20} - FAILED: {str(e)}")
            results[package_name] = False

    return results


def test_ai_models():
    """Test if AI models can be loaded"""
    print("\n🤖 Testing AI Model Loading...")
    print("=" * 40)

    try:
        from sentence_transformers import SentenceTransformer

        # Test lightweight model loading
        print("📥 Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)

        print(f"✅ Model loaded successfully")
        print(f"✅ Embedding generated: shape {embedding.shape}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return False


def test_chromadb():
    """Test ChromaDB functionality"""
    print("\n🗄️ Testing ChromaDB...")
    print("=" * 40)

    try:
        import chromadb

        # Create a temporary client
        client = chromadb.Client()

        # Create a test collection
        collection = client.create_collection("test_collection")

        # Add a test document
        collection.add(
            documents=["This is a test document"],
            metadatas=[{"source": "test"}],
            ids=["test_id"]
        )

        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=1
        )

        print("✅ ChromaDB client created")
        print("✅ Collection created and queried")
        print(f"✅ Query result: {len(results['documents'][0])} documents found")

        return True

    except Exception as e:
        print(f"❌ ChromaDB test failed: {str(e)}")
        return False


def test_file_processing():
    """Test file processing capabilities"""
    print("\n📄 Testing File Processing...")
    print("=" * 40)

    # Test text processing
    try:
        sample_text = "This is a sample resume text for testing text extraction."

        # Test text cleaning (basic)
        import re
        cleaned_text = re.sub(r'\s+', ' ', sample_text)

        print("✅ Text processing - OK")

    except Exception as e:
        print(f"❌ Text processing failed: {str(e)}")
        return False

    # Test PDF processing
    try:
        import PyPDF2
        print("✅ PDF processing capability - Available")
    except ImportError:
        print("⚠️ PDF processing - Not available (PyPDF2 not installed)")

    # Test Word document processing
    try:
        import docx
        print("✅ Word document processing - Available")
    except ImportError:
        print("⚠️ Word document processing - Not available (python-docx not installed)")

    return True


def test_streamlit():
    """Test Streamlit functionality"""
    print("\n🌐 Testing Streamlit...")
    print("=" * 40)

    try:
        import streamlit as st

        # Test basic streamlit functions (these won't display in terminal)
        print("✅ Streamlit imported successfully")
        print("💡 To test Streamlit fully, run: streamlit run app.py")

        return True

    except Exception as e:
        print(f"❌ Streamlit test failed: {str(e)}")
        return False


def test_project_structure():
    """Test project directory structure"""
    print("\n📁 Testing Project Structure...")
    print("=" * 40)

    try:
        from pathlib import Path

        # Expected directories
        expected_dirs = ['data', 'models', 'utils']
        project_root = Path.cwd()

        for dir_name in expected_dirs:
            dir_path = project_root / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/ directory - OK")
            else:
                print(f"⚠️ {dir_name}/ directory - Missing (will be created automatically)")

        # Check for main files
        expected_files = ['app.py', 'config.py', 'requirements.txt']
        for file_name in expected_files:
            file_path = project_root / file_name
            if file_path.exists():
                print(f"✅ {file_name} - OK")
            else:
                print(f"❌ {file_name} - Missing")

        return True

    except Exception as e:
        print(f"❌ Project structure test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and provide summary"""
    print("🚀 AI Resume Job Matcher - Setup Test")
    print("=" * 50)

    test_results = []

    # Run individual tests
    import_results = test_imports()
    test_results.append(('Package Imports', all(import_results.values())))

    ai_result = test_ai_models()
    test_results.append(('AI Models', ai_result))

    db_result = test_chromadb()
    test_results.append(('ChromaDB', db_result))

    file_result = test_file_processing()
    test_results.append(('File Processing', file_result))

    streamlit_result = test_streamlit()
    test_results.append(('Streamlit', streamlit_result))

    structure_result = test_project_structure()
    test_results.append(('Project Structure', structure_result))

    # Print summary
    print("\n📊 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} - {status}")
        if result:
            passed += 1

    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 Next steps:")
        print("1. Run: streamlit run app.py")
        print("2. Open your browser to test the app")
        print("3. Start building your AI features!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print("\n🔧 Common fixes:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check your virtual environment is activated")
        print("3. Verify Python version compatibility")


def show_system_info():
    """Display system information"""
    print("\n💻 SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual Environment: Active")
    else:
        print("⚠️ Virtual Environment: Not detected")

    # Show installed packages count
    try:
        import pkg_resources
        installed_packages = [d for d in pkg_resources.working_set]
        print(f"Installed Packages: {len(installed_packages)}")
    except:
        print("Installed Packages: Unable to determine")


if __name__ == "__main__":
    show_system_info()
    run_all_tests()