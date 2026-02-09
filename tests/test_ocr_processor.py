""
Tests for the OCR Processor module.
"""
import os
import sys
import pytest
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ocr_processor import OCRProcessor, create_ocr_processor

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / 'data'
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Create a simple test image with text
def create_test_image(text: str, output_path: Path, size=(800, 600)):
    """Create a test image with the given text."""
    # Create a white background
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    
    # Add text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)  # Black text
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate text position (centered)
    x = (size[0] - text_width) // 2
    y = (size[1] + text_height) // 2
    
    # Put text on image
    cv2.putText(
        img, text, (x, y), 
        font, font_scale, font_color, thickness, cv2.LINE_AA
    )
    
    # Save the image
    cv2.imwrite(str(output_path), img)
    return img

# Fixtures
@pytest.fixture(scope="module")
def test_image_path():
    """Create a test image and return its path."""
    test_text = "TEST OCR PROCESSOR"
    img_path = TEST_DATA_DIR / "test_ocr.png"
    create_test_image(test_text, img_path)
    yield img_path
    # Cleanup
    if img_path.exists():
        img_path.unlink()

# Tests
def test_ocr_processor_initialization():
    """Test OCR processor initialization."""
    # Test with default parameters
    ocr = OCRProcessor()
    assert ocr is not None
    
    # Test with specific language
    ocr = OCRProcessor(languages=['en', 'fr'])
    assert 'en' in ocr.languages
    assert 'fr' in ocr.languages
    
    # Test with GPU flag
    ocr = OCRProcessor(use_gpu=False)
    assert ocr.use_gpu is False

@patch('easyocr.Reader')
def test_easyocr_initialization(mock_reader):
    """Test EasyOCR initialization."""
    mock_reader.return_value = MagicMock()
    
    # Test successful initialization
    ocr = OCRProcessor()
    assert ocr.easyocr_reader is not None
    
    # Test initialization failure
    mock_reader.side_effect = ImportError("EasyOCR not available")
    ocr = OCRProcessor()
    assert ocr.easyocr_reader is None

def test_process_image_file(test_image_path):
    """Test processing an image from file path."""
    ocr = OCRProcessor()
    success, message, results = ocr.process_image(test_image_path)
    
    assert success is True
    assert "Success" in message
    assert len(results) > 0
    
    # Check if the test text is in the results
    found = any("TEST OCR" in res['text'] for res in results)
    assert found, "Test text not found in OCR results"

def test_process_image_numpy_array(test_image_path):
    """Test processing an image from numpy array."""
    # Load the test image as numpy array
    img = cv2.imread(str(test_image_path))
    assert img is not None
    
    ocr = OCRProcessor()
    success, message, results = ocr.process_image(img)
    
    assert success is True
    assert "Success" in message
    assert len(results) > 0

def test_process_with_invalid_image():
    """Test processing with an invalid image."""
    ocr = OCRProcessor()
    
    # Test with non-existent file
    success, message, results = ocr.process_image("nonexistent.png")
    assert success is False
    assert "not found" in message
    
    # Test with invalid numpy array
    success, message, results = ocr.process_image(np.array([1, 2, 3]))
    assert success is False
    assert "Unsupported" in message

@patch('pytesseract.image_to_data')
def test_tesseract_fallback(mock_tesseract):
    """Test fallback to Tesseract when EasyOCR is not available."""
    # Mock Tesseract response
    mock_tesseract.return_value = {
        'level': [1, 2, 3, 4, 5],
        'text': ['', 'TEST', 'OCR', 'FALLBACK', ''],
        'left': [0, 10, 100, 200, 0],
        'top': [0, 20, 20, 20, 0],
        'width': [100, 50, 60, 100, 0],
        'height': [100, 30, 30, 30, 0],
        'conf': ['-1', '95', '90', '85', '-1']
    }
    
    # Create OCR processor with EasyOCR disabled
    with patch('ocr_processor.OCRProcessor._check_tesseract_installed', return_value=True):
        ocr = OCRProcessor()
        ocr.easyocr_reader = None  # Simulate EasyOCR not available
        
        # Process test image
        success, message, results = ocr.process_image("dummy.png")
        
        assert success is True
        assert len(results) == 3  # Should have 3 text elements (excluding empty and low confidence)
        assert any('TEST' in res['text'] for res in results)
        assert all(res['engine'] == 'tesseract' for res in results)

def test_batch_processing(test_image_path):
    """Test batch processing of multiple images."""
    # Create a second test image
    img_path_2 = TEST_DATA_DIR / "test_ocr_2.png"
    create_test_image("SECOND TEST IMAGE", img_path_2)
    
    try:
        ocr = OCRProcessor()
        results = ocr.batch_process([test_image_path, img_path_2])
        
        assert len(results) == 2
        assert str(test_image_path) in results
        assert str(img_path_2) in results
        
        for img_path, (success, message, ocr_results) in results.items():
            assert success is True
            assert len(ocr_results) > 0
    finally:
        # Cleanup
        if img_path_2.exists():
            img_path_2.unlink()

def test_create_ocr_processor():
    """Test the factory function to create an OCR processor."""
    # Test successful creation
    ocr = create_ocr_processor(languages=['en', 'fr'])
    assert ocr is not None
    assert 'en' in ocr.languages
    assert 'fr' in ocr.languages
    
    # Test with invalid parameters
    with patch('ocr_processor.OCRProcessor.__init__', side_effect=Exception("Test error")):
        ocr = create_ocr_processor()
        assert ocr is None
