# 📰 News Authenticator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/yourusername/news-authenticator/graphs/commit-activity)

<div align="center">
  <img src="static/logo.png" alt="News Authenticator Logo" width="200"/>
  
  🔍 *Verify news authenticity with the power of AI*
</div>

## 🌟 Features

- 🤖 AI-powered news verification using advanced LLM models
- 📊 Real-time news fetching using Google News API
- 🔍 Comprehensive fact-checking through multiple sources
- 🌐 User-friendly web interface
- ⚡ Fast and efficient processing
- 📱 Responsive design for all devices

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository
```bash
https://github.com/ShreyashDabhade/News-Authenticator.git
cd news-authenticator
```

2. Create and activate virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
set GEMINI_API_KEY=your_gemini_api_key
set SERPER_API_KEY=your_serper_api_key
```

5. Run the application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to start using the application.

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **AI/ML**: 
  - Google Gemini AI
  - CrewAI Framework
- **News API**: Google News API
- **Search**: Serper Dev API

## 📖 How It Works

1. User inputs a news article or statement
2. System fetches related news from multiple sources
3. AI agents analyze and cross-reference the information
4. Results are presented with confidence scores and supporting evidence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Google Gemini AI for providing the AI capabilities
- CrewAI framework for AI agent orchestration
- Google News API for real-time news data

---

<div align="center">
  Made with ❤️ for truth in journalism
  
  [Report Bug](https://github.com/yourusername/news-authenticator/issues) · [Request Feature](https://github.com/yourusername/news-authenticator/issues)
</div>
