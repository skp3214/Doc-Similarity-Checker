# **📄 Document Similarity Checker**

🔍 A **modern web application** for comparing two documents and determining their similarity percentage using advanced AI algorithms. Supports PDF, DOC, DOCX, and TXT files with a beautiful glassmorphism UI design.

---

## ✨ Features

### 📄 **Document Comparison**
✅ **Upload Two Documents** - Support for PDF, DOC, DOCX, and TXT files
✅ **AI-Powered Analysis** - Advanced natural language processing using spaCy
✅ **Similarity Scoring** - Get accurate percentage similarity between documents
✅ **Text Extraction** - Automatic text extraction from various file formats
✅ **Real-time Results** - Instant similarity analysis and scoring

### 🎨 **Modern UI/UX**
✅ **Glassmorphism Design** - Beautiful frosted glass effects
✅ **Dark/Light Mode** - Automatic theme detection with manual toggle
✅ **Responsive Design** - Works perfectly on all devices
✅ **Smooth Animations** - Modern transitions and hover effects
✅ **Gradient Cards** - Visual differentiation with multiple gradient styles

### 👤 **User System**
✅ **User Registration & Login** - Secure authentication system
✅ **Profile Management** - User dashboard and settings
✅ **Session Management** - Secure user sessions

---

## 🛠️ Tech Stack

### **Backend**
- **Django** - Python web framework
- **spaCy** - Natural language processing
- **NLTK** - Text processing toolkit
- **scikit-learn** - Machine learning algorithms

### **Frontend**
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with custom properties
- **JavaScript** - Interactive theme management
- **Font Awesome** - Beautiful icons

### **AI & Analysis**
- **TF-IDF Vectorization** - Text similarity analysis
- **Cosine Similarity** - Document comparison algorithm
- **Text Preprocessing** - Tokenization and lemmatization

---

## ✨ ScreenShots

![Screenshot_4-9-2025_13420_doc-similarity-checker onrender com](https://github.com/user-attachments/assets/fe3ed047-6e1d-4e99-8c21-8fdce93e71de)
![Screenshot_4-9-2025_134639_doc-similarity-checker onrender com](https://github.com/user-attachments/assets/728af020-3560-402e-b405-9fda31490692)
![Screenshot_4-9-2025_134758_doc-similarity-checker onrender com](https://github.com/user-attachments/assets/13b4f01b-d246-4111-8921-c06e1f5bff3b)


## 🚀 How It Works

1. **📤 Upload Documents** - Select two files to compare (PDF, DOC, DOCX, or TXT)
2. **🤖 AI Analysis** - Advanced algorithms process and analyze the text content
3. **📊 Get Results** - Receive similarity percentage and detailed analysis
4. **🎨 Modern UI** - Enjoy the beautiful interface with theme switching

---

## 📋 Requirements

- **Python 3.8+**
- **Django 4.0+**
- **spaCy** with English model
- **NLTK** data packages
- **PyPDF2** for PDF processing
- **python-docx** for Word document processing

---

## ⚡ Quick Start

### **Installation**
```bash
# Clone the repository
git clone https://github.com/skp3214/document-similarity-checker.git
cd document-similarity-checker

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### **Setup**
```bash
# Navigate to project directory
cd Doc_Scanner_Matcher

# Run migrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser

# Start development server
python manage.py runserver
```

### **Access**
Open your browser and go to: **http://127.0.0.1:8000/**

---

## 📖 Usage

### **For Users**
1. **🏠 Home Page** - Landing page with modern design
2. **📤 Upload Documents** - Select two files to compare
3. **📊 View Results** - See similarity percentage and analysis
4. **👤 Profile** - Manage your account and settings
5. **🌙 Theme Toggle** - Switch between light and dark modes

### **Supported File Formats**
- 📄 **PDF** - Portable Document Format
- 📝 **DOC/DOCX** - Microsoft Word documents
- 📃 **TXT** - Plain text files

### **AI Analysis Features**
- 🔍 **Text Extraction** - Automatic content extraction
- 🧠 **Semantic Analysis** - Understanding document meaning
- 📈 **Similarity Scoring** - Percentage-based comparison
- 🎯 **Content Matching** - Advanced text comparison algorithms

---

## 🎨 UI Features

### **Modern Design Elements**
- **Glassmorphism** - Frosted glass effects with backdrop blur
- **Gradient Cards** - Multiple gradient styles for visual appeal
- **Smooth Animations** - Hover effects and page transitions
- **Responsive Layout** - Optimized for all screen sizes

### **Theme System**
- **🌞 Light Mode** - Clean, bright interface
- **🌙 Dark Mode** - Easy on the eyes with modern aesthetics
- **Auto Detection** - Respects system preferences
- **Manual Toggle** - One-click theme switching

---

## 📬 Contact & Support

For questions, feedback, or contributions:

📧 **Email**: spsm1818@gmail.com  
🐙 **GitHub**: [skp3214](https://github.com/skp3214)  

### **🐛 Bug Reports & Feature Requests**
- Use [GitHub Issues](https://github.com/skp3214/document-similarity-checker/issues) for bug reports
- Feature requests and UI/UX suggestions are welcome!
- Pull requests for improvements are encouraged

---

## 🏷️ Project Status

### **✅ Current Version: v2.0**
- 🎨 **Modern UI Complete** - Glassmorphism design with dark/light mode
- 🔧 **Fully Functional** - Document comparison working perfectly
- 📱 **Mobile Responsive** - Optimized for all screen sizes
- ♿ **Accessible** - WCAG compliant design
- 🚀 **Production Ready** - Optimized for deployment

### **🔮 Future Enhancements**
- 🌐 **Multi-language Support** (i18n)
- 📊 **Advanced Analytics** with comparison history
- 🔄 **Batch Processing** for multiple document pairs
- 📱 **Progressive Web App** (PWA) features
- 🎯 **API Endpoints** for third-party integrations

---

### 🚀 **Happy Coding!** 😊

*Built with ❤️ using Django, modern CSS, and AI-powered NLP*
