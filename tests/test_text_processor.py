""
Tests for the Text Processing module.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from text_processor import TextProcessor, TextBlock

# Sample test data
SAMPLE_TEXT = """
  John Doe
  CEO, Example Inc.
  
  johndoe@example.com
  (123) 456-7890
  https://example.com
  
  123 Business St, Suite 100
  San Francisco, CA 94107
"""

# Fixtures
@pytest.fixture
def text_processor():
    """Create a TextProcessor instance for testing."""
    return TextProcessor(language='en')

def test_clean_text_basic(text_processor):
    """Test basic text cleaning."""
    # Test whitespace normalization
    assert text_processor.clean_text("  hello   world  ") == "hello world"
    
    # Test special character handling
    assert text_processor.clean_text("hello—world") == "hello-world"
    
    # Test Unicode normalization
    assert text_processor.clean_text("café") == "café"

def test_clean_text_names(text_processor):
    """Test cleaning of names."""
    # Test name cleaning
    assert text_processor.clean_text("jOhN dOe", is_name=True) == "John Doe"
    assert text_processor.clean_text("O'REILLY", is_name=True) == "O'Reilly"
    assert text_processor.clean_text("maria-garcia", is_name=True) == "Maria-Garcia"

def test_extract_contact_info(text_processor):
    """Test extraction of contact information."""
    # Test phone numbers
    test_text = "Call me at (123) 456-7890 or 123-456-7890"
    result = text_processor.extract_contact_info(test_text)
    assert len(result['phones']) == 2
    assert result['phones'][0]['value'] == '(123) 456-7890'
    assert result['phones'][1]['value'] == '123-456-7890'
    
    # Test emails
    test_text = "Email me at test@example.com or contact@test.org"
    result = text_processor.extract_contact_info(test_text)
    assert len(result['emails']) == 2
    assert 'test@example.com' in [e['value'] for e in result['emails']]
    
    # Test URLs
    test_text = "Visit https://example.com or http://test.org"
    result = text_processor.extract_contact_info(test_text)
    assert len(result['urls']) == 2
    assert 'https://example.com' in [u['value'] for u in result['urls']]

def test_process_ocr_results(text_processor):
    """Test processing of OCR results."""
    # Mock OCR results
    ocr_results = [
        {
            'text': '  John Doe  ',
            'bounding_box': [(10, 20), (100, 20), (100, 40), (10, 40)],
            'confidence': 0.95,
            'engine': 'easyocr'
        },
        {
            'text': 'CEO, Example Inc.',
            'bounding_box': [(10, 50), (200, 50), (200, 70), (10, 70)],
            'confidence': 0.90,
            'engine': 'easyocr'
        }
    ]
    
    # Process results
    text_blocks = text_processor.process_ocr_results(ocr_results, image_size=(500, 500))
    
    # Verify results
    assert len(text_blocks) == 2
    assert text_blocks[0].cleaned_text == 'John Doe'
    assert text_blocks[1].cleaned_text == 'CEO, Example Inc.'
    assert text_blocks[0].bounding_box == [(0.02, 0.04), (0.2, 0.04), (0.2, 0.08), (0.02, 0.08)]
    assert text_blocks[0].confidence == 0.95

def test_merge_text_blocks(text_processor):
    """Test merging of text blocks."""
    # Create test text blocks
    blocks = [
        TextBlock(
            text='Hello',
            cleaned_text='Hello',
            bounding_box=[(10, 20), (50, 20), (50, 30), (10, 30)],
            confidence=0.9
        ),
        TextBlock(
            text='World',
            cleaned_text='World',
            bounding_box=[(60, 20), (100, 20), (100, 30), (60, 30)],
            confidence=0.95
        )
    ]
    
    # Merge blocks
    merged = text_processor.merge_text_blocks(blocks, max_line_gap=0.1, max_word_gap=0.1)
    
    # Verify merging
    assert len(merged) == 1
    assert merged[0].cleaned_text == 'Hello World'
    assert merged[0].confidence == 0.9  # Should take the minimum confidence
    assert merged[0].bounding_box == [(10, 20), (100, 20), (100, 30), (10, 30)]

def test_post_process(text_processor):
    """Test post-processing of text blocks."""
    # Create test text blocks
    blocks = [
        TextBlock(
            text='John Doe',
            cleaned_text='John Doe',
            bounding_box=[(10, 20), (100, 20), (100, 40), (10, 40)],
            confidence=0.95,
            metadata={'source': 'test'}
        ),
        TextBlock(
            text='CEO, Example Inc.',
            cleaned_text='CEO, Example Inc.',
            bounding_box=[(10, 50), (200, 50), (200, 70), (10, 70)],
            confidence=0.90,
            metadata={'source': 'test'}
        ),
        TextBlock(
            text='johndoe@example.com',
            cleaned_text='johndoe@example.com',
            bounding_box=[(10, 80), (200, 80), (200, 100), (10, 100)],
            confidence=0.98,
            metadata={'source': 'test'}
        )
    ]
    
    # Process blocks
    result = text_processor.post_process(blocks)
    
    # Verify results
    assert result['full_text'] == 'John Doe\nCEO, Example Inc.\njohndoe@example.com'
    assert len(result['entities']['names']) == 1
    assert result['entities']['names'][0]['value'] == 'John Doe'
    assert len(result['entities']['titles']) == 1
    assert 'CEO' in result['entities']['titles'][0]['value']
    assert len(result['entities']['emails']) == 1
    assert 'johndoe@example.com' in [e['value'] for e in result['entities']['emails']]
    assert result['metadata']['block_count'] == 3

def test_edge_cases(text_processor):
    """Test edge cases and error handling."""
    # Test empty input
    assert text_processor.clean_text("") == ""
    assert text_processor.clean_text(None) == ""
    
    # Test with None/empty OCR results
    assert text_processor.process_ocr_results([]) == []
    assert text_processor.merge_text_blocks([]) == []
    
    # Test with None bounding box
    blocks = [
        TextBlock(text="Test", cleaned_text="Test", bounding_box=None, confidence=0.9)
    ]
    assert text_processor.merge_text_blocks(blocks) == blocks  # Should return as-is
    
    # Test post-process with empty blocks
    empty_result = text_processor.post_process([])
    assert empty_result['full_text'] == ''
    assert all(len(v) == 0 for v in empty_result['entities'].values())

@patch('text_processor.ftfy.fix_text')
def test_clean_text_encoding_issues(mock_fix_text, text_processor):
    """Test handling of encoding issues in text cleaning."""
    # Test with broken Unicode
    mock_fix_text.return_value = "fixed text"
    assert text_processor.clean_text("broken\xc3\x28text") == "fixed text"
    mock_fix_text.assert_called_once()

def test_phone_number_extraction(text_processor):
    """Test extraction of various phone number formats."""
    test_cases = [
        ("Call me at (123) 456-7890", ["(123) 456-7890"]),
        ("My number is 123.456.7890", ["123.456.7890"]),
        ("Cell: 123-456-7890 Work: 987-654-3210", ["123-456-7890", "987-654-3210"]),
        ("International +1 (123) 456-7890", ["+1 (123) 456-7890"]),
        ("No phone here", []),
    ]
    
    for text, expected in test_cases:
        result = text_processor.extract_contact_info(text)
        assert len(result['phones']) == len(expected)
        assert all(phone['value'] in expected for phone in result['phones'])

def test_email_extraction(text_processor):
    """Test extraction of email addresses."""
    test_cases = [
        ("Email me at test@example.com", ["test@example.com"]),
        ("Contact: user.name+tag@sub.domain.co.uk", ["user.name+tag@sub.domain.co.uk"]),
        ("No email here", []),
        ("Invalid email@", []),
    ]
    
    for text, expected in test_cases:
        result = text_processor.extract_contact_info(text)
        assert len(result['emails']) == len(expected)
        assert all(email['value'] in expected for email in result['emails'])

def test_url_extraction(text_processor):
    """Test extraction of URLs."""
    test_cases = [
        ("Visit https://example.com", ["https://example.com"]),
        ("http://test.org and https://sub.domain.co.uk/path?q=query", 
         ["http://test.org", "https://sub.domain.co.uk/path?q=query"]),
        ("No URL here", []),
        ("Invalid http://", []),
    ]
    
    for text, expected in test_cases:
        result = text_processor.extract_contact_info(text)
        assert len(result['urls']) == len(expected)
        assert all(url['value'] in expected for url in result['urls'])
