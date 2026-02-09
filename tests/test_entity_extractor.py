"""
Tests for the Entity Extraction module.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import spacy
from spacy.language import Language

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from entity_extractor import EntityExtractor, Entity, create_entity_extractor

# Sample test data
SAMPLE_TEXT = """
John Doe
Chief Technology Officer
Acme Technologies Inc.

Email: john.doe@acme.com
Phone: (123) 456-7890
Website: https://acme.com

123 Business Street, Suite 100
San Francisco, CA 94107
"""

# Fixtures
@pytest.fixture
def entity_extractor():
    """Create an EntityExtractor instance for testing."""
    # Use a small English model for testing
    return EntityExtractor(language='en', model_name='en_core_web_sm')

@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        Entity(
            text="John Doe",
            label="PERSON",
            start_char=1,
            end_char=9,
            confidence=0.95,
            source="ml"
        ),
        Entity(
            text="Acme Technologies Inc.",
            label="ORG",
            start_char=30,
            end_char=52,
            confidence=0.90,
            source="ml"
        ),
        Entity(
            text="john.doe@acme.com",
            label="EMAIL",
            start_char=70,
            end_char=88,
            confidence=0.99,
            source="regex"
        )
    ]

def test_entity_extractor_initialization(entity_extractor):
    """Test entity extractor initialization."""
    assert entity_extractor is not None
    assert entity_extractor.language == 'en'
    assert entity_extractor.model_name == 'en_core_web_sm'

def test_extract_entities(entity_extractor):
    """Test entity extraction from text."""
    entities = entity_extractor.extract_entities(SAMPLE_TEXT)
    
    # We should have at least some entities
    assert len(entities) > 0
    
    # Check that we have the expected entity types
    entity_labels = {e.label for e in entities}
    assert 'PERSON' in entity_labels or 'ORG' in entity_labels or 'EMAIL' in entity_labels

def test_extract_structured_data(entity_extractor, sample_entities):
    """Test structured data extraction."""
    result = entity_extractor.extract_structured_data(SAMPLE_TEXT, sample_entities)
    
    # Check basic structure
    assert 'name' in result
    assert 'company' in result
    assert 'emails' in result
    
    # Check extracted values
    assert result['name'] == 'John Doe'
    assert 'Acme' in result['company']
    assert 'john.doe@acme.com' in result['emails']

def test_merge_overlapping_entities(entity_extractor):
    """Test merging of overlapping entities."""
    # Create overlapping entities
    entities = [
        Entity("John", "PERSON", 0, 4, 0.9, "ml"),
        Entity("John Doe", "PERSON", 0, 8, 0.95, "ml"),
        Entity("Doe", "PERSON", 5, 8, 0.85, "ml")
    ]
    
    merged = entity_extractor._merge_overlapping_entities(entities, "John Doe")
    
    # Should be merged into a single entity
    assert len(merged) == 1
    assert merged[0].text == "John Doe"
    assert merged[0].confidence >= 0.9  # Should have high confidence

def test_post_process_entities(entity_extractor):
    """Test post-processing of entities."""
    # Create entities with whitespace
    entities = [
        Entity("  John Doe  ", "PERSON", 0, 12, 0.9, "ml"),
        Entity("Acme  ", "ORG", 15, 20, 0.85, "ml")
    ]
    
    processed = entity_extractor._post_process_entities(entities, "  John Doe  works at Acme  ")
    
    # Whitespace should be trimmed
    assert processed[0].text == "John Doe"
    assert processed[1].text == "Acme"
    
    # Positions should be adjusted
    assert processed[0].start_char == 2
    assert processed[0].end_char == 10

def test_extract_contact_info(entity_extractor):
    """Test extraction of contact information."""
    text = "Contact me at test@example.com or (123) 456-7890"
    doc = entity_extractor.nlp(text)
    entities = entity_extractor._extract_contact_info(text)
    
    # Should find both email and phone
    assert len(entities) == 2
    assert any(e.label == 'EMAIL' for e in entities)
    assert any(e.label == 'PHONE' for e in entities)

def test_create_entity_extractor():
    """Test factory function for creating entity extractor."""
    extractor = create_entity_extractor(language='en')
    assert extractor is not None
    assert isinstance(extractor, EntityExtractor)

@patch('spacy.load')
def test_load_spacy_model(mock_load):
    """Test loading of spaCy model with fallback."""
    # Mock the spaCy load function
    mock_nlp = MagicMock()
    mock_load.return_value = mock_nlp
    
    # Test successful load
    extractor = EntityExtractor(model_name='en_core_web_sm')
    assert extractor.nlp is not None
    
    # Test fallback when model is not found
    mock_load.side_effect = OSError("Model not found")
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        extractor = EntityExtractor(model_name='en_core_web_sm')
        assert extractor.nlp is not None

def test_entity_to_dict(sample_entities):
    """Test conversion of Entity to dictionary."""
    entity_dict = sample_entities[0].to_dict()
    assert isinstance(entity_dict, dict)
    assert entity_dict['text'] == 'John Doe'
    assert entity_dict['label'] == 'PERSON'
    assert 'metadata' in entity_dict

def test_extract_with_custom_patterns(tmp_path):
    """Test extraction with custom patterns."""
    # Create a temporary patterns file
    patterns = [
        {"label": "SKILL", "pattern": "machine learning"},
        {"label": "SKILL", "pattern": "natural language processing"}
    ]
    
    patterns_file = tmp_path / "patterns.json"
    with open(patterns_file, 'w') as f:
        json.dump(patterns, f)
    
    # Create extractor with custom patterns
    extractor = EntityExtractor(patterns_file=str(patterns_file))
    
    # Test extraction with custom patterns
    text = "I specialize in machine learning and natural language processing."
    entities = extractor.extract_entities(text)
    
    # Should find both skills
    skill_texts = [e.text for e in entities if e.label == 'SKILL']
    assert 'machine learning' in skill_texts
    assert 'natural language processing' in skill_texts

def test_extract_structured_data_edge_cases(entity_extractor):
    """Test edge cases in structured data extraction."""
    # Test with empty text
    result = entity_extractor.extract_structured_data("")
    assert result['name'] is None
    assert result['emails'] == []
    
    # Test with None
    result = entity_extractor.extract_structured_data(None)
    assert result['name'] is None
    
    # Test with no entities
    result = entity_extractor.extract_structured_data("Some random text")
    assert result['name'] is None

def test_name_extraction_from_email(entity_extractor):
    """Test extraction of name from email address."""
    text = "Contact: john.doe@example.com"
    entities = [
        Entity("john.doe@example.com", "EMAIL", 9, 29, 0.99, "regex")
    ]
    
    result = entity_extractor.extract_structured_data(text, entities)
    assert result['name'] == 'John Doe'
    assert 'john.doe@example.com' in result['emails']

if __name__ == "__main__":
    pytest.main([__file__])
