#!/bin/bash

# RAG Installation Script for Law Q&A System
# This script installs all RAG dependencies including FAISS

echo "🚀 Installing RAG Dependencies for Law Q&A System..."
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "📌 Python version: $(python3 --version)"

# Install core dependencies first
echo ""
echo "📦 Installing core dependencies..."
pip3 install --upgrade pip
pip3 install streamlit>=1.31.0
pip3 install groq>=0.4.0
pip3 install tavily-python>=0.3.0
pip3 install python-dotenv>=1.0.1
pip3 install pypdf>=3.0.0
pip3 install python-docx>=1.1.0
pip3 install langchain>=0.1.0
pip3 install langchain-core>=0.1.0
pip3 install langchain-community>=0.0.29
pip3 install sentence-transformers>=2.2.2
pip3 install numpy>=1.24.0
pip3 install requests>=2.31.0
pip3 install beautifulsoup4>=4.12.0
pip3 install lxml>=4.9.0

echo ""
echo "📦 Installing FAISS (Vector Database)..."
echo ""

# Check if SWIG is installed
if command -v swig &> /dev/null; then
    echo "✅ SWIG is installed. Installing FAISS from source..."
    pip3 install faiss-cpu --no-cache-dir
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "⚠️  SWIG not found. Attempting to install via Homebrew..."
    if command -v brew &> /dev/null; then
        echo "Installing SWIG via Homebrew..."
        brew install swig
        pip3 install faiss-cpu --no-cache-dir
    else
        echo "❌ Homebrew not found. Please install SWIG manually:"
        echo "   1. Install Homebrew: https://brew.sh"
        echo "   2. Run: brew install swig"
        echo "   3. Run: pip3 install faiss-cpu"
        echo ""
        echo "OR install FAISS via conda:"
        echo "   conda install -c pytorch faiss-cpu"
        exit 1
    fi
else
    echo "⚠️  SWIG not found. Please install it:"
    echo "   Ubuntu/Debian: sudo apt-get install swig"
    echo "   Or use conda: conda install -c pytorch faiss-cpu"
    pip3 install faiss-cpu || echo "⚠️  FAISS installation failed. Install SWIG first."
fi

echo ""
echo "✅ Verifying installation..."
python3 -c "
try:
    import langchain
    import sentence_transformers
    print('✅ LangChain: OK')
    print('✅ Sentence Transformers: OK')
except ImportError as e:
    print(f'❌ Error: {e}')
    exit(1)

try:
    import faiss
    print('✅ FAISS: OK')
except ImportError:
    print('⚠️  FAISS: Not installed (RAG will work but vector search will be limited)')
    print('   To install: Install SWIG, then run: pip3 install faiss-cpu')
    print('   OR use conda: conda install -c pytorch faiss-cpu')
"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Get your API keys:"
echo "      - Groq: https://console.groq.com/"
echo "      - Tavily: https://tavily.com/"
echo ""
echo "   2. Run the app:"
echo "      streamlit run app.py"
echo ""
