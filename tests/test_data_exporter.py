"""
Tests for the Data Export module.
"""
import os
import sys
import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_exporter import BusinessCardExporter, create_exporter

# Sample test data
SAMPLE_CARDS = [
    {
        'name': 'John Doe',
        'job_title': 'Chief Technology Officer',
        'company': 'Acme Technologies Inc.',
        'emails': ['john.doe@acme.com', 'jdoe@example.com'],
        'phones': ['(123) 456-7890', '+1 (987) 654-3210'],
        'urls': ['https://acme.com', 'https://linkedin.com/in/johndoe'],
        'address': '123 Business St, San Francisco, CA 94107',
        'notes': 'Met at Tech Conference 2023',
        'source_image': 'card1.jpg',
        'extraction_date': '2023-06-15 14:30:00'
    },
    {
        'name': 'Jane Smith',
        'job_title': 'Senior Product Manager',
        'company': 'InnovateCorp',
        'emails': ['jane.smith@innovatecorp.com'],
        'phones': ['(555) 123-4567'],
        'urls': ['https://innovatecorp.com'],
        'address': '456 Innovation Ave, Boston, MA 02108',
        'notes': 'Follow up next week',
        'source_image': 'card2.jpg',
        'extraction_date': '2023-06-16 10:15:00'
    }
]

# Fixtures
@pytest.fixture
temp_dir(tmp_path):
    """Create a temporary directory for test exports."""
    return tmp_path / "exports"

@pytest.fixture
def exporter(temp_dir):
    """Create a BusinessCardExporter instance with a temporary output directory."""
    return BusinessCardExporter(output_dir=temp_dir)

def test_exporter_initialization(temp_dir):
    """Test exporter initialization with custom output directory."""
    exporter = BusinessCardExporter(output_dir=temp_dir)
    assert exporter.output_dir == temp_dir
    assert temp_dir.exists()

def test_export_to_excel(exporter, temp_dir):
    """Test exporting business cards to Excel."""
    # Test with default filename
    output_path = exporter.export_to_excel(SAMPLE_CARDS)
    assert os.path.exists(output_path)
    assert output_path.endswith('.xlsx')
    
    # Verify the file was created in the correct directory
    assert str(temp_dir) in output_path
    
    # Verify the Excel file can be read
    df = pd.read_excel(output_path)
    assert len(df) == len(SAMPLE_CARDS)
    assert df['name'].tolist() == [card['name'] for card in SAMPLE_CARDS]

def test_export_with_custom_filename(exporter, temp_dir):
    """Test exporting with a custom filename."""
    custom_name = "my_business_cards.xlsx"
    output_path = exporter.export_to_excel(
        SAMPLE_CARDS,
        filename=custom_name,
        include_timestamp=False
    )
    
    assert os.path.basename(output_path) == custom_name
    assert os.path.exists(output_path)

def test_export_empty_data(exporter):
    """Test exporting empty data raises an error."""
    with pytest.raises(ValueError):
        exporter.export_to_excel([])

def test_export_with_additional_fields(exporter, temp_dir):
    """Test exporting data with additional custom fields."""
    custom_cards = [
        {
            'name': 'Test User',
            'company': 'Test Corp',
            'custom_field': 'Custom Value',
            'another_field': 123
        }
    ]
    
    output_path = exporter.export_to_excel(custom_cards, include_timestamp=False)
    df = pd.read_excel(output_path)
    
    # Check that custom fields are included
    assert 'custom_field' in df.columns
    assert 'another_field' in df.columns
    assert df.iloc[0]['custom_field'] == 'Custom Value'
    assert df.iloc[0]['another_field'] == 123

def test_export_multiple_sheets(exporter, temp_dir):
    """Test exporting multiple sheets to a single Excel file."""
    sheets_data = {
        'Tech Industry': [SAMPLE_CARDS[0]],
        'Product Management': [SAMPLE_CARDS[1]]
    }
    
    output_path = exporter.export_multiple_sheets(
        sheets_data,
        filename='multiple_sheets.xlsx',
        include_timestamp=False
    )
    
    assert os.path.exists(output_path)
    
    # Verify both sheets exist and have the correct data
    with pd.ExcelFile(output_path) as xls:
        assert set(xls.sheet_names) == {'Tech Industry', 'Product Management'}
        
        df_tech = pd.read_excel(xls, 'Tech Industry')
        assert len(df_tech) == 1
        assert df_tech.iloc[0]['name'] == 'John Doe'
        
        df_pm = pd.read_excel(xls, 'Product Management')
        assert len(df_pm) == 1
        assert df_pm.iloc[0]['name'] == 'Jane Smith'

def test_export_with_special_characters(exporter, temp_dir):
    """Test exporting data with special characters."""
    special_card = [{
        'name': 'José González',
        'company': 'Café & Co.',
        'notes': 'Special chars: áéíóú ñÑ çÇ',
        'emails': ['test@example.com']
    }]
    
    output_path = exporter.export_to_excel(special_card, include_timestamp=False)
    df = pd.read_excel(output_path)
    
    assert df.iloc[0]['name'] == 'José González'
    assert 'Café & Co.' in df.iloc[0]['company']
    assert 'áéíóú' in df.iloc[0]['notes']

def test_create_exporter_factory(temp_dir):
    """Test the create_exporter factory function."""
    exporter = create_exporter(output_dir=temp_dir)
    assert isinstance(exporter, BusinessCardExporter)
    assert exporter.output_dir == temp_dir

@patch('pandas.DataFrame.to_excel')
def test_export_error_handling(mock_to_excel, exporter):
    """Test error handling during export."""
    # Simulate an error during Excel export
    mock_to_excel.side_effect = Exception("Test error")
    
    with pytest.raises(Exception) as exc_info:
        exporter.export_to_excel(SAMPLE_CARDS)
    
    assert "Test error" in str(exc_info.value)

def test_export_with_none_values(exporter, temp_dir):
    """Test exporting data with None values."""
    cards_with_none = [
        {
            'name': 'Test User',
            'company': None,
            'emails': ['test@example.com'],
            'phones': None,
            'notes': None
        }
    ]
    
    output_path = exporter.port_to_excel(cards_with_none, include_timestamp=False)
    df = pd.read_excel(output_path)
    
    # Check that None values are handled correctly
    assert pd.isna(df.iloc[0]['company'])
    assert pd.isna(df.iloc[0]['phones'])
    assert pd.isna(df.iloc[0]['notes'])
    assert df.iloc[0]['name'] == 'Test User'

def test_export_with_large_data(exporter, temp_dir):
    """Test exporting a large number of business cards."""
    # Create a large dataset
    large_dataset = []
    for i in range(100):  # 100 cards
        large_dataset.append({
            'name': f'User {i}',
            'company': f'Company {i % 10}',
            'emails': [f'user{i}@example.com'],
            'phones': [f'({i:03}) 555-{i:04}']
        })
    
    output_path = exporter.export_to_excel(
        large_dataset,
        filename='large_dataset.xlsx',
        include_timestamp=False
    )
    
    # Verify the file was created and has the correct number of rows
    df = pd.read_excel(output_path)
    assert len(df) == 100
    assert df['name'].tolist() == [f'User {i}' for i in range(100)]

if __name__ == "__main__":
    pytest.main([__file__])
