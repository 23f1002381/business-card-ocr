"""
Text Processing Module

This module handles cleaning, normalizing, and preparing OCR output for entity extraction.
"""
import re
import logging
import unicodedata
from typing import Dict, List, Optional, Tuple, Union, Any
import string
import ftfy
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class TextBlock:
    """Represents a block of processed text with metadata."""
    text: str
    cleaned_text: str
    bounding_box: Optional[List[Tuple[float, float]]] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the TextBlock to a dictionary."""
        return {
            'text': self.text,
            'cleaned_text': self.cleaned_text,
            'bounding_box': self.bounding_box,
            'confidence': self.confidence,
            'language': self.language,
            'metadata': self.metadata or {}
        }

class TextProcessor:
    """
    Handles text cleaning, normalization, and preparation for entity extraction.
    """
    
    # Common noise patterns in OCR output
    NOISE_PATTERNS = [
        (r'^\s*[\*_\-~=+#]{2,}\s*$', ''),  # Lines with only symbols
        (r'^\s*[0-9]+\s*$', ''),            # Lines with only numbers
        (r'^[^a-zA-Z0-9]{3,}$', ''),         # Lines with only special chars
        (r'\s+', ' '),                       # Multiple whitespace
        (r'\n\s*\n', '\n'),              # Multiple newlines
    ]
    
    # Common replacements for OCR errors
    OCR_REPLACEMENTS = [
        # Common OCR misreads
        (r'[|]', 'I'),
        (r'[1l]', 'I'),
        (r'[0]', 'O'),
        (r'[5]', 'S'),
        (r'[`\'\"]', ''),  # Remove quotes that might be OCR artifacts
    ]
    
    # Common business card patterns
    PHONE_PATTERNS = [
        # International format: +1 (123) 456-7890
        r'\+?\d{1,4}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
        # Standard format: (123) 456-7890 or 123-456-7890
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        # Without separators: 1234567890
        r'\d{10,}',
    ]
    
    EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    URL_PATTERN = r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)'
    
    def __init__(self, language: str = 'en'):
        """
        Initialize the text processor.
        
        Args:
            language: Default language for text processing
        """
        self.language = language
        self.compiled_patterns = {
            'noise': [(re.compile(pattern), repl) for pattern, repl in self.NOISE_PATTERNS],
            'ocr_replace': [(re.compile(pattern), repl) for pattern, repl in self.OCR_REPLACEMENTS],
            'phone': re.compile('|'.join(f'({pattern})' for pattern in self.PHONE_PATTERNS)),
            'email': re.compile(self.EMAIL_PATTERN),
            'url': re.compile(self.URL_PATTERN),
        }
    
    def clean_text(self, text: str, is_name: bool = False) -> str:
        """
        Clean and normalize text from OCR output.
        
        Args:
            text: Input text to clean
            is_name: Whether the text is a person's name (applies special handling)
            
        Returns:
            Cleaned and normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Fix encoding issues and common Unicode problems
        text = ftfy.fix_text(text)
        
        # Normalize Unicode (NFKC normalization)
        text = unicodedata.normalize('NFKC', text)
        
        # Apply OCR-specific replacements
        for pattern, repl in self.compiled_patterns['ocr_replace']:
            text = pattern.sub(repl, text)
        
        # Handle names specially (preserve capitalization)
        if is_name:
            # Remove any non-name characters but preserve spaces and hyphens
            text = re.sub(r'[^\p{L}\s\-\']', ' ', text, flags=re.UNICODE)
            # Clean up whitespace
            text = ' '.join(text.strip().split())
            # Title case for names (handles hyphenated names)
            text = ' '.join(word.capitalize() for word in text.split())
            return text
        
        # Standard cleaning for non-name text
        
        # Remove control characters
        text = ''.join(char for char in text if char in string.printable or char.isspace())
        
        # Apply noise patterns
        for pattern, repl in self.compiled_patterns['noise']:
            text = re.sub(pattern, repl, text, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove duplicate whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_contact_info(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract common contact information from text.
        
        Args:
            text: Input text to extract information from
            
        Returns:
            Dictionary containing lists of extracted entities by type
        """
        results = {
            'phones': [],
            'emails': [],
            'urls': [],
        }
        
        # Extract phone numbers
        for match in self.compiled_patterns['phone'].finditer(text):
            phone = match.group(0).strip()
            results['phones'].append({
                'value': phone,
                'start': match.start(),
                'end': match.end(),
                'type': 'phone',
                'source': 'regex'
            })
        
        # Extract email addresses
        for match in self.compiled_patterns['email'].finditer(text):
            email = match.group(0).strip()
            results['emails'].append({
                'value': email,
                'start': match.start(),
                'end': match.end(),
                'type': 'email',
                'source': 'regex'
            })
        
        # Extract URLs
        for match in self.compiled_patterns['url'].finditer(text):
            url = match.group(0).strip()
            results['urls'].append({
                'value': url,
                'start': match.start(),
                'end': match.end(),
                'type': 'url',
                'source': 'regex'
            })
        
        return results
    
    def process_ocr_results(
        self, 
        ocr_results: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None
    ) -> List[TextBlock]:
        """
        Process raw OCR results into clean, structured text blocks.
        
        Args:
            ocr_results: List of OCR results from the OCRProcessor
            image_size: Optional (width, height) of the source image for normalization
            
        Returns:
            List of processed TextBlock objects
        """
        text_blocks = []
        
        for i, result in enumerate(ocr_results):
            if not result.get('text', '').strip():
                continue
                
            # Clean the text
            cleaned_text = self.clean_text(result['text'])
            if not cleaned_text:
                continue
            
            # Normalize bounding box coordinates if image size is provided
            bbox = result.get('bounding_box')
            if bbox and image_size:
                width, height = image_size
                bbox = [
                    (x / width, y / height) 
                    for x, y in bbox
                ]
            
            # Create text block
            text_block = TextBlock(
                text=result['text'],
                cleaned_text=cleaned_text,
                bounding_box=bbox,
                confidence=result.get('confidence'),
                language=self.language,
                metadata={
                    'source_engine': result.get('engine', 'unknown'),
                    'source_index': i,
                }
            )
            
            text_blocks.append(text_block)
        
        return text_blocks
    
    def merge_text_blocks(
        self, 
        text_blocks: List[TextBlock],
        max_line_gap: float = 0.1,
        max_word_gap: float = 0.05
    ) -> List[TextBlock]:
        """
        Merge text blocks that are likely part of the same line or paragraph.
        
        Args:
            text_blocks: List of TextBlock objects
            max_line_gap: Maximum vertical gap (as fraction of image height) to consider as same line
            max_word_gap: Maximum horizontal gap (as fraction of image width) to consider as same word
            
        Returns:
            List of merged TextBlock objects
        """
        if not text_blocks:
            return []
        
        # Sort text blocks by vertical position, then by horizontal position
        sorted_blocks = sorted(
            text_blocks,
            key=lambda b: (
                min(y for x, y in b.bounding_box) if b.bounding_box else 0,
                min(x for x, y in b.bounding_box) if b.bounding_box else 0
            )
        )
        
        merged_blocks = []
        current_line = []
        
        def process_line():
            """Process and merge blocks in the current line."""
            if not current_line:
                return
                
            # Sort blocks in the line from left to right
            line_blocks = sorted(
                current_line,
                key=lambda b: min(x for x, y in b.bounding_box) if b.bounding_box else 0
            )
            
            # Merge blocks that are close to each other
            merged_text = []
            merged_bbox = None
            min_confidence = 1.0
            
            for i, block in enumerate(line_blocks):
                if i > 0 and block.bounding_box and line_blocks[i-1].bounding_box:
                    # Calculate gap between previous and current block
                    prev_right = max(x for x, y in line_blocks[i-1].bounding_box)
                    curr_left = min(x for x, y in block.bounding_box)
                    gap = curr_left - prev_right
                    
                    # Add space if blocks are not too far apart
                    if gap > 0 and gap < max_word_gap * 100:  # Assuming 100 is the reference width
                        merged_text.append(' ')
                
                merged_text.append(block.cleaned_text)
                
                # Update merged bounding box
                if block.bounding_box:
                    if merged_bbox is None:
                        merged_bbox = [
                            (x, y) for x, y in block.bounding_box
                        ]
                    else:
                        # Expand the bounding box to include this block
                        min_x = min(merged_bbox[0][0], min(x for x, y in block.bounding_box))
                        min_y = min(merged_bbox[0][1], min(y for x, y in block.bounding_box))
                        max_x = max(merged_bbox[2][0], max(x for x, y in block.bounding_box))
                        max_y = max(merged_bbox[2][1], max(y for x, y in block.bounding_box))
                        
                        merged_bbox = [
                            (min_x, min_y),
                            (max_x, min_y),
                            (max_x, max_y),
                            (min_x, max_y)
                        ]
                
                # Track minimum confidence
                if block.confidence is not None:
                    min_confidence = min(min_confidence, block.confidence)
            
            # Create merged block
            merged_block = TextBlock(
                text=''.join(merged_text),
                cleaned_text=' '.join(block.cleaned_text for block in line_blocks),
                bounding_box=merged_bbox,
                confidence=min_confidence if min_confidence < 1.0 else None,
                language=self.language,
                metadata={
                    'merged_from': [block.metadata for block in line_blocks],
                    'line_number': len(merged_blocks) + 1
                }
            )
            
            merged_blocks.append(merged_block)
        
        # Group text blocks into lines
        for i, block in enumerate(sorted_blocks):
            if not block.bounding_box:
                # If no bounding box, treat as a new line
                process_line()
                current_line = [block]
                continue
                
            if not current_line:
                current_line.append(block)
                continue
                
            # Calculate vertical position (average y-coordinate of top edge)
            current_y = sum(y for x, y in block.bounding_box) / len(block.bounding_box)
            
            # Get the last block in the current line
            last_block = current_line[-1]
            if not last_block.bounding_box:
                process_line()
                current_line = [block]
                continue
                
            last_y = sum(y for x, y in last_block.bounding_box) / len(last_block.bounding_box)
            
            # If vertical position is similar, add to current line
            if abs(current_y - last_y) < max_line_gap * 100:  # Assuming 100 is the reference height
                current_line.append(block)
            else:
                # Start a new line
                process_line()
                current_line = [block]
        
        # Process the last line
        process_line()
        
        return merged_blocks
    
    def post_process(self, text_blocks: List[TextBlock]) -> Dict[str, Any]:
        """
        Post-process text blocks to extract structured information.
        
        Args:
            text_blocks: List of TextBlock objects
            
        Returns:
            Dictionary containing structured information
        """
        result = {
            'full_text': '\n'.join(block.cleaned_text for block in text_blocks),
            'text_blocks': [block.to_dict() for block in text_blocks],
            'entities': {
                'names': [],
                'emails': [],
                'phones': [],
                'urls': [],
                'companies': [],
                'titles': [],
                'addresses': [],
            },
            'metadata': {
                'language': self.language,
                'block_count': len(text_blocks),
                'avg_confidence': (
                    sum(b.confidence for b in text_blocks if b.confidence is not None) / 
                    len([b for b in text_blocks if b.confidence is not None])
                    if any(b.confidence is not None for b in text_blocks)
                    else None
                )
            }
        }
        
        # Extract basic contact information
        for block in text_blocks:
            # Extract emails, phones, and URLs
            contact_info = self.extract_contact_info(block.cleaned_text)
            
            # Add to results
            result['entities']['emails'].extend([
                {'value': email['value'], 'source_block': block.to_dict()}
                for email in contact_info['emails']
            ])
            
            result['entities']['phones'].extend([
                {'value': phone['value'], 'source_block': block.to_dict()}
                for phone in contact_info['phones']
            ])
            
            result['entities']['urls'].extend([
                {'value': url['value'], 'source_block': block.to_dict()}
                for url in contact_info['urls']
            ])
            
            # Simple heuristic for names and titles
            # This will be enhanced in the entity extraction module
            text = block.cleaned_text
            if not any(c.isdigit() for c in text) and len(text.split()) <= 4:
                # Could be a name or title
                if any(title in text.lower() for title in ['ceo', 'cto', 'manager', 'director', 'president']):
                    result['entities']['titles'].append({
                        'value': text,
                        'source_block': block.to_dict()
                    })
                elif len(text.split()) >= 2:  # Likely a name if 2+ words
                    result['entities']['names'].append({
                        'value': text,
                        'source_block': block.to_dict()
                    })
            
            # Simple company detection (all caps, contains common company suffixes)
            if (text.isupper() or any(word.istitle() for word in text.split())) and \
               any(suffix in text.lower() for suffix in ['inc', 'llc', 'ltd', 'corp', 'co\.', 'gmbh']):
                result['entities']['companies'].append({
                    'value': text,
                    'source_block': block.to_dict()
                })
        
        return result
