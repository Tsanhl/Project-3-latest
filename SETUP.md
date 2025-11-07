# 🚀 Setup Guide - Law Q&A System with RAG

This guide will help you install all dependencies and run the Law Q&A application.

## 📋 Prerequisites

- Python 3.8-3.11 (recommended) or Python 3.12+
- pip (Python package manager)
- Internet connection (for downloading packages and API access)

## 🔧 Installation Methods

### Method 1: Automated Installation (Recommended)

Use our installation script:

```bash
# Make script executable (if not already)
chmod +x install_rag.sh

# Run installation
./install_rag.sh
```

### Method 2: Manual Installation

#### Step 1: Install Core Dependencies

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

#### Step 2: Install FAISS (Vector Database)

**Option A: Using Conda (Easiest - Recommended)**

```bash
# Install conda if you don't have it: https://docs.conda.io/en/latest/miniconda.html
conda install -c pytorch faiss-cpu
```

**Option B: Install SWIG First, Then FAISS**

**On macOS:**
```bash
# Install Homebrew if needed: https://brew.sh
brew install swig
pip3 install faiss-cpu
```

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install swig
pip3 install faiss-cpu
```

**On Windows:**
1. Download SWIG from: https://swig.org/download.html
2. Extract and add to PATH
3. Run: `pip install faiss-cpu`

**Option C: Use Pre-built Wheel (Python 3.8-3.11 only)**

```bash
# For macOS ARM64 (M1/M2/M3)
pip3 install https://github.com/facebookresearch/faiss/releases/download/v1.7.4/faiss-cpu-1.7.4-cp311-cp311-macosx_10_9_arm64.whl

# For macOS Intel
pip3 install https://github.com/facebookresearch/faiss/releases/download/v1.7.4/faiss-cpu-1.7.4-cp311-cp311-macosx_10_9_x86_64.whl
```

**Option D: Skip FAISS (Limited RAG Features)**

If you can't install FAISS, the app will still work but RAG features will be limited. The system will use web search and free law databases instead.

## ✅ Verify Installation

Test if everything is installed correctly:

```bash
python3 -c "
import streamlit, groq, tavily, langchain, sentence_transformers
print('✅ Core dependencies: OK')

try:
    import faiss
    print('✅ FAISS: OK - Full RAG features enabled')
except ImportError:
    print('⚠️  FAISS: Not installed - RAG will use fallback methods')
    print('   To enable full RAG: Install SWIG and run: pip3 install faiss-cpu')
"
```

## 🔑 Get API Keys (FREE)

### 1. Groq API Key

1. Visit [Groq Console](https://console.groq.com/)
2. Sign up (free, no credit card required)
3. Go to **API Keys** section
4. Click **"Create API Key"**
5. Copy your key (starts with `gsk_`)

**Free Tier:** 14,400 requests/day

### 2. Tavily API Key

1. Visit [Tavily AI](https://tavily.com/)
2. Sign up (free)
3. Go to Dashboard → **API Keys**
4. Generate API key (starts with `tvly-`)
5. Copy your key

**Free Tier:** 1,000 searches/month

## 🚀 Run the Application

### Option 1: Using Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Then run:

```bash
streamlit run app.py
```

### Option 2: Enter Keys in UI

Just run the app and enter keys in the sidebar:

```bash
streamlit run app.py
```

The application will open at: **http://localhost:8501**

## 📁 Project Structure

```
Project 3/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── install_rag.sh           # Automated installation script
├── SETUP.md                 # This file
├── readme.md                # Project documentation
├── Law resouces/            # Your law documents (PDFs, DOCX)
│   ├── Biolaw copy/
│   ├── Business law copy/
│   ├── Commercial Law copy/
│   └── ... (all subfolders automatically indexed)
└── .env                     # API keys (create this file)
```

## 🔍 Troubleshooting

### Issue: "SWIG not found" when installing FAISS

**Solution:**
- **macOS:** `brew install swig` then `pip3 install faiss-cpu`
- **Ubuntu/Debian:** `sudo apt-get install swig` then `pip3 install faiss-cpu`
- **Windows:** Download SWIG from https://swig.org, add to PATH, then `pip install faiss-cpu`
- **Alternative:** Use conda: `conda install -c pytorch faiss-cpu`

### Issue: "RAG libraries not fully installed" warning

**Solution:**
1. Check if FAISS is installed: `python3 -c "import faiss; print('OK')"`
2. If not, follow FAISS installation steps above
3. If FAISS still fails, the app will work with limited RAG (uses web search instead)

### Issue: "ModuleNotFoundError: No module named 'langchain'"

**Solution:**
```bash
pip3 install langchain langchain-core langchain-community
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue: Python 3.13 and FAISS won't install

**Solution:**
- FAISS may not have pre-built wheels for Python 3.13 yet
- Option 1: Use Python 3.11 or 3.12 (recommended)
- Option 2: Install SWIG and build from source
- Option 3: Use conda which has pre-built FAISS packages

## 📚 What Each Component Does

- **Streamlit**: Web interface framework
- **Groq**: Fast LLM inference (Llama 3 models)
- **Tavily**: Real-time web search API
- **LangChain**: RAG framework for document processing
- **FAISS**: Vector database for similarity search (needs SWIG)
- **Sentence Transformers**: Embedding models for text similarity

## 🎯 Quick Start Checklist

- [ ] Python 3.8-3.11 installed (or 3.12+)
- [ ] Run `./install_rag.sh` OR manually install dependencies
- [ ] Get Groq API key: https://console.groq.com/
- [ ] Get Tavily API key: https://tavily.com/
- [ ] Create `.env` file OR enter keys in UI
- [ ] Place law documents in `Law resouces/` folder (optional)
- [ ] Run: `streamlit run app.py`
- [ ] Open browser: http://localhost:8501

## 🆘 Need Help?

1. Check the [Troubleshooting](#-troubleshooting) section above
2. Verify all dependencies: `python3 -c "import streamlit, groq, tavily, langchain"`
3. Check Python version: `python3 --version` (3.8-3.11 recommended)
4. Review error messages in terminal where Streamlit is running

## 🌐 Running on a Server

To make the app accessible from other devices on your network:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```

Then access from other devices using: `http://YOUR_IP_ADDRESS:8501`

## 🐳 Docker Alternative

If you prefer Docker, see the `Dockerfile` and `docker-compose.yml` files for containerized deployment.

---

**🎉 Ready to go!** Once installed, run `streamlit run app.py` and start asking law questions!
