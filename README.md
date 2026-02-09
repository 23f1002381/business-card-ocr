# Business Card OCR Application

A comprehensive solution for digitizing business cards using OCR and machine learning. This application allows users to upload images of business cards, extract contact information, and export the data to Excel.

## Features

- **Image Processing**: Preprocess and enhance business card images for better OCR accuracy
- **OCR Integration**: Supports multiple OCR engines (EasyOCR and Tesseract)
- **Entity Extraction**: Extracts names, job titles, companies, and contact information
- **Data Export**: Export extracted data to Excel with professional formatting
- **User-Friendly Interface**: Simple and intuitive web interface built with Streamlit

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/business-card-ocr.git
   cd business-card-ocr
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (required for fallback OCR)
   - **Windows**: Download and install from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

5. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Open the app** in your browser at `http://localhost:8501`

3. **Upload business card images** using the sidebar

4. **Process the images** to extract information

5. **Review and edit** the extracted information

6. **Export the data** to Excel

## Project Structure

```
business-card-ocr/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── src/                 # Source code
│   ├── __init__.py
│   ├── image_processing.py  # Image preprocessing
│   ├── ocr_processor.py     # OCR processing
│   ├── text_processor.py    # Text processing
│   ├── entity_extractor.py  # Entity extraction
│   └── data_exporter.py     # Data export to Excel
├── tests/               # Unit tests
│   ├── __init__.py
│   ├── test_image_processing.py
│   ├── test_ocr_processor.py
│   ├── test_text_processor.py
│   ├── test_entity_extractor.py
│   └── test_data_exporter.py
├── uploads/             # Uploaded business card images
├── processed/           # Processed images
└── exports/             # Exported Excel files
```

## Configuration

You can configure the application by creating a `.env` file in the project root:

```
# OCR settings
OCR_ENGINE=easyocr  # or 'tesseract'
LANGUAGES=en,fr     # comma-separated list of languages

# Image processing
ENHANCE_CONTRAST=true
DENOISE=true

# Output settings
DEFAULT_EXPORT_FORMAT=xlsx
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the awesome web framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) and [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- [spaCy](https://spacy.io/) for natural language processing
- [OpenCV](https://opencv.org/) for image processing
