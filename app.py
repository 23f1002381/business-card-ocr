""
Business Card OCR Application

A Streamlit-based web interface for extracting and managing business card information.
"""
import os
import sys
import logging
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import base64
import json
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from src.image_processing import process_business_card
from src.ocr_processor import create_ocr_processor
from src.text_processor import TextProcessor
from src.entity_extractor import create_entity_extractor
from src.data_exporter import create_exporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Business Card OCR",
    page_icon="📇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
EXPORT_DIR = Path("exports")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create necessary directories
for directory in [UPLOAD_DIR, PROCESSED_DIR, EXPORT_DIR]:
    directory.mkdir(exist_ok=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'processed_cards' not in st.session_state:
    st.session_state.processed_cards = {}
if 'current_card_index' not in st.session_state:
    st.session_state.current_card_index = 0

# Initialize processors (lazy loading)
@st.cache_resource
def get_ocr_processor():
    """Get the OCR processor instance."""
    return create_ocr_processor(languages=['en'])

@st.cache_resource
def get_text_processor():
    """Get the text processor instance."""
    return TextProcessor(language='en')

@st.cache_resource
def get_entity_extractor():
    """Get the entity extractor instance."""
    return create_entity_extractor(language='en')

def save_uploaded_file(uploaded_file) -> Path:
    """Save uploaded file to the upload directory."""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_image(image_path: Path) -> Dict[str, Any]:
    """Process a single business card image."""
    try:
        # Step 1: Image preprocessing
        logger.info(f"Processing image: {image_path}")
        original_img, processed_img = process_business_card(
            image_path,
            output_path=PROCESSED_DIR / f"processed_{image_path.name}",
            save_processed=True
        )
        
        if original_img is None or processed_img is None:
            raise ValueError("Failed to process image")
        
        # Step 2: OCR processing
        ocr_processor = get_ocr_processor()
        success, message, ocr_results = ocr_processor.process_image(processed_img)
        
        if not success:
            raise ValueError(f"OCR processing failed: {message}")
        
        # Step 3: Text processing
        text_processor = get_text_processor()
        text_blocks = text_processor.process_ocr_results(ocr_results, image_size=original_img.shape[:2])
        merged_blocks = text_processor.merge_text_blocks(text_blocks)
        processed_data = text_processor.post_process(merged_blocks)
        
        # Step 4: Entity extraction
        entity_extractor = get_entity_extractor()
        structured_data = entity_extractor.extract_structured_data(
            '\n'.join(b.cleaned_text for b in merged_blocks)
        )
        
        # Combine results
        result = {
            'image_path': str(image_path),
            'processed_image_path': str(PROCESSED_DIR / f"processed_{image_path.name}"),
            'raw_text': '\n'.join(b.cleaned_text for b in merged_blocks),
            'processed_data': processed_data,
            'structured_data': structured_data,
            'extraction_date': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return {
            'image_path': str(image_path),
            'error': str(e),
            'status': 'error'
        }

def display_business_card(card_data: Dict[str, Any]):
    """Display the processed business card data."""
    if card_data.get('status') == 'error':
        st.error(f"Error processing card: {card_data.get('error')}")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display the processed image
        if os.path.exists(card_data['processed_image_path']):
            st.image(
                card_data['processed_image_path'],
                caption="Processed Business Card",
                use_column_width=True
            )
        else:
            st.warning("Processed image not found")
        
        # Display raw text
        with st.expander("View Extracted Text"):
            st.text(card_data['raw_text'])
    
    with col2:
        # Display structured data
        st.subheader("Extracted Information")
        
        # Basic info
        cols = st.columns(2)
        with cols[0]:
            st.text_input("Name", 
                         value=card_data['structured_data'].get('name', ''), 
                         key=f"name_{card_data['image_path']}")
        
        with cols[1]:
            st.text_input("Job Title", 
                         value=card_data['structured_data'].get('job_title', ''), 
                         key=f"title_{card_data['image_path']}")
        
        st.text_input("Company", 
                     value=card_data['structured_data'].get('company', ''), 
                     key=f"company_{card_data['image_path']}")
        
        # Contact info
        st.subheader("Contact Information")
        
        # Emails
        emails = card_data['structured_data'].get('emails', [])
        email = st.text_input("Email", 
                             value=emails[0] if emails else '', 
                             key=f"email_{card_data['image_path']}")
        
        # Phones
        phones = card_data['structured_data'].get('phones', [])
        phone = st.text_input("Phone", 
                             value=phones[0] if phones else '', 
                             key=f"phone_{card_data['image_path']}")
        
        # Website
        urls = card_data['structured_data'].get('urls', [])
        website = st.text_input("Website", 
                               value=urls[0] if urls else '', 
                               key=f"website_{card_data['image_path']}")
        
        # Address
        address = st.text_area("Address", 
                              value=card_data['structured_data'].get('address', ''), 
                              key=f"address_{card_data['image_path']}")
        
        # Notes
        notes = st.text_area("Notes", 
                           value=card_data['structured_data'].get('notes', ''), 
                           key=f"notes_{card_data['image_path']}")
        
        # Update the card data with any user edits
        card_data['structured_data'].update({
            'name': st.session_state.get(f"name_{card_data['image_path']}", ''),
            'job_title': st.session_state.get(f"title_{card_data['image_path']}", ''),
            'company': st.session_state.get(f"company_{card_data['image_path']}", ''),
            'emails': [st.session_state.get(f"email_{card_data['image_path']}", '')],
            'phones': [st.session_state.get(f"phone_{card_data['image_path']}", '')],
            'urls': [st.session_state.get(f"website_{card_data['image_path']}", '')],
            'address': st.session_state.get(f"address_{card_data['image_path']}", ''),
            'notes': st.session_state.get(f"notes_{card_data['image_path']}", '')
        })

def export_to_excel():
    """Export all processed cards to an Excel file."""
    if not st.session_state.processed_cards:
        st.warning("No processed cards to export")
        return None
    
    try:
        # Prepare data for export
        export_data = []
        for card in st.session_state.processed_cards.values():
            if card.get('status') != 'success':
                continue
                
            export_data.append({
                'name': card['structured_data'].get('name', ''),
                'job_title': card['structured_data'].get('job_title', ''),
                'company': card['structured_data'].get('company', ''),
                'email': ', '.join(card['structured_data'].get('emails', [])),
                'phone': ', '.join(card['structured_data'].get('phones', [])),
                'website': ', '.join(card['structured_data'].get('urls', [])),
                'address': card['structured_data'].get('address', ''),
                'notes': card['structured_data'].get('notes', ''),
                'source_image': card['image_path'],
                'extraction_date': card['extraction_date']
            })
        
        # Create exporter and export to Excel
        exporter = create_exporter(output_dir=EXPORT_DIR)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = exporter.export_to_excel(
            export_data,
            filename=f"business_cards_{timestamp}.xlsx",
            include_timestamp=False
        )
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {str(e)}")
        st.error(f"Error exporting data: {str(e)}")
        return None

def main():
    """Main application function."""
    st.title("📇 Business Card OCR")
    st.markdown("Upload business card images to extract contact information.")
    
    # Sidebar for file upload and actions
    with st.sidebar:
        st.header("Upload Business Cards")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose business card images",
            type=list(ALLOWED_EXTENSIONS),
            accept_multiple_files=True
        )
        
        # Process button
        if st.button("Process Images", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one image")
            else:
                with st.spinner("Processing images..."):
                    for uploaded_file in uploaded_files:
                        if uploaded_file.size > MAX_FILE_SIZE:
                            st.warning(f"File {uploaded_file.name} is too large. Max size is 10MB.")
                            continue
                        
                        try:
                            # Save and process the uploaded file
                            file_path = save_uploaded_file(uploaded_file)
                            
                            # Process the image
                            result = process_image(file_path)
                            
                            # Store the result
                            if result['status'] == 'success':
                                st.session_state.processed_cards[file_path.name] = result
                                st.session_state.uploaded_files.append(file_path.name)
                                st.success(f"Processed: {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                        
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Export button
        st.divider()
        st.header("Export Data")
        
        if st.session_state.processed_cards:
            if st.button("Export to Excel", type="primary"):
                output_path = export_to_excel()
                if output_path:
                    st.success(f"Successfully exported to {output_path}")
                    
                    # Provide download link
                    with open(output_path, "rb") as f:
                        bytes_data = f.read()
                    
                    st.download_button(
                        label="Download Excel File",
                        data=bytes_data,
                        file_name=os.path.basename(output_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("Process some business cards to enable export")
        
        # Display processing stats
        st.divider()
        st.header("Statistics")
        st.metric("Processed Cards", f"{len(st.session_state.processed_cards)}")
    
    # Main content area
    if not st.session_state.processed_cards:
        st.info("Upload and process business card images to get started.")
    else:
        # Display processed cards in tabs
        tab_titles = [f"Card {i+1}" for i in range(len(st.session_state.processed_cards))]
        tabs = st.tabs(tab_titles)
        
        for i, (file_name, card_data) in enumerate(st.session_state.processed_cards.items()):
            with tabs[i]:
                display_business_card(card_data)

if __name__ == "__main__":
    main()
