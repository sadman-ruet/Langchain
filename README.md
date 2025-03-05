# PDF to QnA

## Overview

PDF to QnA is a powerful tool that allows you to extract and generate question-answer pairs from any PDF document. Built with Langchain and Hugging Face, this project enables users to process PDFs efficiently and retrieve meaningful insights using NLP.

## Features

- **Upload PDF**: Easily upload any PDF file for processing.
- **Q&A**: Ask anything from book.
- **Interactive Interface**: User-friendly UI powered by Streamlit.
- **Efficient Processing**: Utilizes Langchain for natural language processing and extraction.

## Demo

Try out the live demo here: [Hugging Face Space](https://huggingface.co/spaces/sadman-hasib/Pdf_To_Qna)

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/Pdf_To_Qna.git
cd Pdf_To_Qna
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Upload a PDF, and the app will generate questions and answers based on the content.

## Folder Structure

```
Pdf_To_Qna/
│── app.py            # Main application file
│── requirements.txt  # Required dependencies
│── README.md         # Documentation
```

## Dependencies

- Python 3.x
- Streamlit
- Langchain
- Hugging Face Transformers
- PyMuPDF (for PDF processing)
- OpenAI API (optional for advanced QnA generation)

## Contributing

We welcome contributions! If you’d like to improve this project, feel free to fork the repository, create a feature branch, and submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

Thanks to [Sadman Hasib](https://huggingface.co/spaces/sadman-hasib/Pdf_To_Qna) for the original implementation and Hugging Face for providing powerful NLP models.

