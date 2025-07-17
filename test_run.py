def test_imports():
    """Test if all required packages are installed correctly."""
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")

        import sentence_transformers
        print("✅ Sentence-transformers imported successfully")

        import chromadb
        print("✅ ChromaDB imported successfully")

        import transformers
        print("✅ Transformers imported successfully")

        import torch
        print("✅ PyTorch imported successfully")

        import PyPDF2
        print("✅ PyPDF2 imported successfully")

        import pandas as pd
        print("✅ Pandas imported successfully")

        import numpy as np
        print("✅ NumPy imported successfully")

        print("\n🎉 All packages installed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    test_imports()