#  MCQ Generator
  
MCQ Generator is a Flask-based web application that allows users to generate multiple-choice questions (MCQs) from text input. This tool is useful for educators, students, and content creators looking to automate quiz generation.  

## Features  
✅ Accepts text input via URL, manual entry, or file uploads (PDF/TXT)  
✅ Generates multiple-choice questions dynamically  
✅ Users can select the number of questions to generate  
✅ Displays correct answers for review  
✅ Allows MCQ download as a PDF  
✅ Simple and user-friendly interface  

## Technologies Used  
- Flask (Backend)  
- BeautifulSoup & Requests (Web Scraping)  
- PyPDF2 (PDF Processing)  
- Spacy (NLP Processing)  
- ReportLab (PDF Generation)  
- Bootstrap (Frontend UI)  

## Getting Started  
Clone this repository and follow the installation steps in the README to set up the project locally.  

---

## Table of Contents

- [Features](#features)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
  - [Creating a Virtual Environment](#creating-a-virtual-environment)
  - [Installing Dependencies](#installing-dependencies)
- [Usage](#usage)
- [Deployment](#deployment)
- [Notes](#notes)

## Features

- **MCQ Generation:** Automatically generates MCQs by processing input text, PDFs, or URLs.
- **NLP & Deep Learning:** Uses spaCy for text processing and a TensorFlow/Keras LSTM model to understand sentence structures.
- **PDF Download:** Allows users to download the generated MCQs as a PDF.
- **Responsive UI:** Utilizes Flask-Bootstrap for a clean, responsive web interface.

## Folder Structure

```plaintext
your_project_folder/
├── app.py
├── requirements.txt
├── README.md
├── static/
│   └── style.css
└── templates/
    ├── index.html
    ├── mcqs.html
    └── result.html
```

## Creating a Virtual Environment

It is recommended to create a virtual environment to manage dependencies. To create a virtual environment named `mcqvenv`, run the following commands in your project folder:

### On Windows:

```bash
python -m venv mcqvenv
mcqvenv\Scripts\activate
```

## Installing Dependencies

With the virtual environment activated, install the project dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```
## Usage

### Run the Application:
Ensure your virtual environment is active, then start the Flask app:

```bash
python app.py
```
### Access the App:

Open your web browser and navigate to http://127.0.0.1:5000.

### Generate MCQs:

- Provide input via a URL, manual text, or by uploading PDF/TXT files.
- Select the number of questions to generate.
- Click on Generate MCQs.
-View the generated questions and optionally download them as a PDF.

### View Detailed Results:

- A detailed view is available which shows each question, its options, and the correct answer.

## Deployment

For deployment on platforms such as Render, Railway, or Heroku, make sure you include a Procfile with the following content:

```bash
web: gunicorn app:app
```

### Configure your deployment service to use:

- Build Command: pip install -r requirements.txt
- Start Command: gunicorn app:app

## Notes
- Security: Replace the placeholder app.secret_key in app.py with a strong, random secret key in a production environment.
- Model Training: The LSTM model in this project is initialized using a sample text. For improved performance, consider training the model with a larger and more diverse dataset.
- Dependencies: Ensure that the versions specified in requirements.txt are maintained for compatibility, particularly TensorFlow with NumPy.

**Happy coding!**
