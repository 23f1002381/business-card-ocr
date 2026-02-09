"""
Entity Extraction Module

This module handles the extraction of structured information from processed text,
including names, job titles, companies, and contact information.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.matcher import PhraseMatcher, Matcher
import numpy as np
from pathlib import Path
import json

# Initialize logger
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents an extracted entity with metadata."""
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float
    source: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the entity to a dictionary."""
        return {
            'text': self.text,
            'label': self.label,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'confidence': self.confidence,
            'source': self.source,
            'metadata': self.metadata or {}
        }

class EntityExtractor:
    """
    Extracts entities from text using a combination of rule-based and ML-based approaches.
    """
    
    # Common job title keywords (will be extended)
    JOB_TITLE_KEYWORDS = [
        'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio', 'cpo', 'cso', 'cdo', 'cto',
        'director', 'manager', 'president', 'vp', 'vice president', 'head of',
        'lead', 'senior', 'junior', 'chief', 'officer', 'founder', 'partner',
        'architect', 'engineer', 'developer', 'designer', 'analyst', 'specialist',
        'consultant', 'advisor', 'researcher', 'scientist', 'professor', 'doctor'
    ]
    
    # Common company suffixes
    COMPANY_SUFFIXES = [
        'inc', 'llc', 'ltd', 'corp', 'co', 'llp', 'plc', 'gmbh', 'pty', 'ltd',
        'limited', 'incorporated', 'corporation', 'company', 'group', 'holdings',
        'technologies', 'solutions', 'systems', 'ventures', 'partners', 'associates'
    ]
    
    # Common department names
    DEPARTMENTS = [
        'engineering', 'research', 'development', 'product', 'design', 'marketing',
        'sales', 'hr', 'human resources', 'finance', 'operations', 'it', 'legal',
        'customer support', 'business development', 'r&d', 'quality assurance'
    ]
    
    def __init__(self, language: str = 'en', model_name: str = 'en_core_web_sm'):
        """
        Initialize the entity extractor.
        
        Args:
            language: Language code (e.g., 'en', 'fr')
            model_name: Name of the spaCy model to use
        """
        self.language = language
        self.model_name = model_name
        self.nlp = self._load_spacy_model()
        self._init_matchers()
        self._load_custom_patterns()
    
    def _load_spacy_model(self) -> Language:
        """Load the spaCy language model."""
        try:
            # Try to load the model
            nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except OSError:
            # Download the model if not found
            logger.warning(f"spaCy model '{self.model_name}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", self.model_name])
            nlp = spacy.load(self.model_name)
            logger.info(f"Downloaded and loaded spaCy model: {self.model_name}")
        
        # Add custom pipeline components
        if not nlp.has_pipe("entity_ruler"):
            ruler = nlp.add_pipe("entity_ruler")
            self._add_patterns_to_ruler(ruler)
        
        return nlp
    
    def _init_matchers(self):
        """Initialize the phrase and rule matchers."""
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Add patterns to matchers
        self._add_job_title_patterns()
        self._add_company_patterns()
    
    def _load_custom_patterns(self, patterns_file: Optional[str] = None):
        """Load custom entity patterns from a JSON file."""
        self.custom_patterns = {}
        
        if patterns_file and Path(patterns_file).exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.custom_patterns = json.load(f)
                logger.info(f"Loaded custom patterns from {patterns_file}")
            except Exception as e:
                logger.error(f"Error loading custom patterns: {e}")
    
    def _add_patterns_to_ruler(self, ruler):
        """Add patterns to the entity ruler."""
        # Add job title patterns
        for title in self.JOB_TITLE_KEYWORDS:
            ruler.add_patterns([{"label": "JOB_TITLE", "pattern": title}])
        
        # Add company patterns
        for suffix in self.COMPANY_SUFFIXES:
            ruler.add_patterns([{"label": "ORG", "pattern": suffix}])
        
        # Add custom patterns if available
        if self.custom_patterns:
            ruler.add_patterns(self.custom_patterns)
    
    def _add_job_title_patterns(self):
        """Add patterns for job title detection."""
        # Simple job title patterns
        job_patterns = [
            [{"LOWER": {"in": ["senior", "junior", "lead", "chief"]}, "OP": "?"},
             {"POS": {"in": ["NOUN", "PROPN"]}, "OP": "+"}],
            [{"POS": "PROPN", "OP": "+"}, 
             {"LOWER": {"in": ["manager", "director", "officer"]}}],
            [{"LOWER": "vice"}, {"LOWER": "president"}],
            [{"LOWER": "head"}, {"LOWER": "of"}, {"POS": "NOUN"}]
        ]
        
        for pattern in job_patterns:
            self.matcher.add("JOB_TITLE", [pattern])
    
    def _add_company_patterns(self):
        """Add patterns for company name detection."""
        # Company name patterns
        company_patterns = [
            [{"POS": "PROPN", "OP": "+"}, 
             {"LOWER": {"in": self.COMPANY_SUFFIXES}, "OP": "?"}],
            [{"POS": "PROPN"}, {"LOWER": "&", "OP": "?"}, {"POS": "PROPN"}]
        ]
        
        for pattern in company_patterns:
            self.matcher.add("ORG", [pattern])
    
    def extract_entities(
        self, 
        text: str, 
        use_ml: bool = True,
        use_rules: bool = True,
        merge_overlapping: bool = True
    ) -> List[Entity]:
        """
        Extract entities from text using a combination of methods.
        
        Args:
            text: Input text to extract entities from
            use_ml: Whether to use machine learning-based extraction
            use_rules: Whether to use rule-based extraction
            merge_overlapping: Whether to merge overlapping entities
            
        Returns:
            List of extracted entities
        """
        if not text or not text.strip():
            return []
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract entities using different methods
        entities = []
        
        # 1. ML-based NER (if enabled)
        if use_ml:
            ml_entities = self._extract_ml_entities(doc)
            entities.extend(ml_entities)
        
        # 2. Rule-based extraction (if enabled)
        if use_rules:
            rule_entities = self._extract_rule_based_entities(doc)
            entities.extend(rule_entities)
        
        # 3. Merge overlapping entities
        if merge_overlapping and len(entities) > 1:
            entities = self._merge_overlapping_entities(entities, text)
        
        # 4. Post-process entities
        entities = self._post_process_entities(entities, text)
        
        return entities
    
    def _extract_ml_entities(self, doc: Doc) -> List[Entity]:
        """Extract entities using the ML model."""
        entities = []
        
        for ent in doc.ents:
            # Skip low-confidence entities
            if ent.label_ == 'CARDINAL' and not any(c.isalpha() for c in ent.text):
                continue
                
            # Map spaCy labels to our labels
            label = self._map_entity_label(ent.label_)
            
            # Estimate confidence (spaCy doesn't provide this natively)
            confidence = self._estimate_confidence(ent.text, label)
            
            entities.append(Entity(
                text=ent.text,
                label=label,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=confidence,
                source='ml',
                metadata={
                    'spacy_label': ent.label_,
                    'spacy_label_': ent.label
                }
            ))
        
        return entities
    
    def _extract_rule_based_entities(self, doc: Doc) -> List[Entity]:
        """Extract entities using rule-based methods."""
        entities = []
        
        # 1. Use the matcher to find patterns
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            label = self.nlp.vocab.strings[match_id]
            
            # Skip very short matches (likely false positives)
            if len(span.text.strip()) < 3:
                continue
            
            # Estimate confidence
            confidence = self._estimate_confidence(span.text, label)
            
            entities.append(Entity(
                text=span.text,
                label=label,
                start_char=span.start_char,
                end_char=span.end_char,
                confidence=confidence,
                source='rules',
                metadata={
                    'pattern_id': match_id,
                    'pattern': self._get_pattern_for_match(match_id)
                }
            ))
        
        # 2. Extract email addresses and phone numbers using regex
        entities.extend(self._extract_contact_info(doc.text))
        
        return entities
    
    def _extract_contact_info(self, text: str) -> List[Entity]:
        """Extract contact information using regex patterns."""
        entities = []
        
        # Email addresses
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for match in re.finditer(email_pattern, text):
            entities.append(Entity(
                text=match.group(0),
                label='EMAIL',
                start_char=match.start(),
                end_char=match.end(),
                confidence=0.99,  # High confidence for well-formed emails
                source='regex',
                metadata={'pattern': 'email'}
            ))
        
        # Phone numbers
        phone_patterns = [
            # International format: +1 (123) 456-7890
            r'\+?\d{1,4}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            # Standard format: (123) 456-7890 or 123-456-7890
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            # Without separators: 1234567890
            r'\d{10,}'
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                # Skip if this is part of a larger number (e.g., in an address)
                if not self._is_valid_phone(match.group(0), text, match.start(), match.end()):
                    continue
                
                entities.append(Entity(
                    text=match.group(0),
                    label='PHONE',
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.95,  # High confidence for well-formed phone numbers
                    source='regex',
                    metadata={'pattern': 'phone'}
                ))
        
        return entities
    
    def _is_valid_phone(self, phone: str, text: str, start: int, end: int) -> bool:
        """Check if a matched phone number is valid in its context."""
        # Skip very long numbers (likely not a phone number)
        if len(phone) > 15 and not phone.startswith('+'):
            return False
        
        # Check surrounding characters
        prev_char = text[start-1] if start > 0 else ' '
        next_char = text[end] if end < len(text) else ' '
        
        # Phone numbers are usually surrounded by whitespace or punctuation
        if prev_char.isalnum() or next_char.isalnum():
            return False
            
        return True
    
    def _map_entity_label(self, label: str) -> str:
        """Map spaCy entity labels to our custom labels."""
        label_map = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'LOCATION',
            'LOC': 'LOCATION',
            'FAC': 'FACILITY',
            'PRODUCT': 'PRODUCT',
            'EVENT': 'EVENT',
            'WORK_OF_ART': 'WORK_OF_ART',
            'LAW': 'LAW',
            'LANGUAGE': 'LANGUAGE',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'PERCENT': 'PERCENT',
            'MONEY': 'MONEY',
            'QUANTITY': 'QUANTITY',
            'ORDINAL': 'ORDINAL',
            'CARDINAL': 'CARDINAL',
            'NORP': 'NORP',
            'FAC': 'FACILITY'
        }
        
        return label_map.get(label, 'MISC')
    
    def _estimate_confidence(self, text: str, label: str) -> float:
        """Estimate confidence score for an entity."""
        # Base confidence
        confidence = 0.8
        
        # Adjust based on entity type
        if label in ['PERSON', 'ORG', 'GPE']:
            confidence += 0.1
        
        # Adjust based on text characteristics
        if len(text.split()) > 1:
            confidence += 0.05
        
        # Cap at 0.99 (never 1.0 to indicate some uncertainty)
        return min(0.99, confidence)
    
    def _get_pattern_for_match(self, match_id: int) -> str:
        """Get the pattern that matched for a given match ID."""
        # This is a simplified version - in practice, you'd want to store
        # the patterns in a more accessible way
        return f"pattern_{match_id}"
    
    def _merge_overlapping_entities(
        self, 
        entities: List[Entity], 
        text: str
    ) -> List[Entity]:
        """Merge overlapping or adjacent entities."""
        if not entities:
            return []
        
        # Sort entities by start position
        entities.sort(key=lambda e: (e.start_char, -e.end_char))
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            # If entities overlap or are adjacent
            if entity.start_char <= current.end_char:
                # Merge with current entity
                start = min(current.start_char, entity.start_char)
                end = max(current.end_char, entity.end_char)
                
                # Choose the label with higher confidence
                if entity.confidence > current.confidence:
                    label = entity.label
                    source = entity.source
                else:
                    label = current.label
                    source = current.source
                
                # Calculate combined confidence (weighted average)
                current_len = current.end_char - current.start_char
                entity_len = entity.end_char - entity.start_char
                total_len = current_len + entity_len
                
                confidence = (
                    (current.confidence * current_len) + 
                    (entity.confidence * entity_len)
                ) / total_len
                
                # Create merged entity
                current = Entity(
                    text=text[start:end],
                    label=label,
                    start_char=start,
                    end_char=end,
                    confidence=confidence,
                    source=f"merged({current.source},{entity.source})",
                    metadata={
                        'merged_from': [current.to_dict(), entity.to_dict()],
                        'original_sources': [current.source, entity.source]
                    }
                )
            else:
                # No overlap, add current to merged and move to next
                merged.append(current)
                current = entity
        
        # Add the last entity
        merged.append(current)
        
        return merged
    
    def _post_process_entities(
        self, 
        entities: List[Entity],
        text: str
    ) -> List[Entity]:
        """Post-process entities to improve quality."""
        processed = []
        
        for entity in entities:
            # Skip very short entities (likely noise)
            if len(entity.text.strip()) < 2:
                continue
            
            # Clean up entity text
            cleaned_text = entity.text.strip()
            
            # Skip if text is empty after cleaning
            if not cleaned_text:
                continue
            
            # Update entity with cleaned text
            entity.text = cleaned_text
            
            # Adjust positions if text was trimmed
            if entity.text != entity.text.strip():
                # Find the cleaned text in the original text
                start = text.find(cleaned_text, entity.start_char)
                if start != -1:
                    entity.start_char = start
                    entity.end_char = start + len(cleaned_text)
            
            # Add to processed list
            processed.append(entity)
        
        # Remove duplicates (same text and label)
        seen = set()
        unique_entities = []
        
        for entity in processed:
            key = (entity.text.lower(), entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_structured_data(
        self, 
        text: str,
        entities: Optional[List[Entity]] = None
    ) -> Dict[str, Any]:
        """
        Extract structured data from text and entities.
        
        Args:
            text: The input text
            entities: Optional pre-extracted entities
            
        Returns:
            Dictionary with structured data
        """
        if entities is None:
            entities = self.extract_entities(text)
        
        # Initialize result structure
        result = {
            'name': None,
            'job_title': None,
            'company': None,
            'emails': [],
            'phones': [],
            'urls': [],
            'address': None,
            'other_entities': []
        }
        
        # Categorize entities
        for entity in entities:
            if entity.label == 'PERSON' and not result['name']:
                result['name'] = entity.text
            elif entity.label == 'JOB_TITLE' and not result['job_title']:
                result['job_title'] = entity.text
            elif entity.label == 'ORG' and not result['company']:
                # Try to find the most likely company name
                if not result['company'] or len(entity.text) > len(result['company']):
                    result['company'] = entity.text
            elif entity.label == 'EMAIL':
                result['emails'].append(entity.text)
            elif entity.label == 'PHONE':
                result['phones'].append(entity.text)
            elif entity.label == 'URL':
                result['urls'].append(entity.text)
            elif entity.label in ['STREET', 'CITY', 'STATE', 'ZIP', 'COUNTRY']:
                # Simple address detection (will be improved)
                if not result['address']:
                    result['address'] = entity.text
                else:
                    result['address'] += f", {entity.text}"
            else:
                result['other_entities'].append({
                    'text': entity.text,
                    'label': entity.label,
                    'confidence': entity.confidence
                })
        
        # Clean up results
        if result['emails']:
            result['emails'] = list(set(result['emails']))  # Remove duplicates
        
        if result['phones']:
            result['phones'] = list(set(result['phones']))  # Remove duplicates
        
        if result['urls']:
            result['urls'] = list(set(result['urls']))  # Remove duplicates
        
        # Try to extract name from email if not found
        if not result['name'] and result['emails']:
            email = result['emails'][0]
            name_part = email.split('@')[0]
            
            # Simple heuristic: split on common separators and capitalize
            for sep in ['.', '_', '-']:
                if sep in name_part:
                    name = ' '.join(part.capitalize() for part in name_part.split(sep))
                    result['name'] = name
                    break
        
        return result


def create_entity_extractor(
    language: str = 'en',
    model_name: str = 'en_core_web_sm',
    patterns_file: Optional[str] = None
) -> EntityExtractor:
    """
    Factory function to create an entity extractor.
    
    Args:
        language: Language code (e.g., 'en', 'fr')
        model_name: Name of the spaCy model to use
        patterns_file: Path to custom patterns file (JSON)
        
    Returns:
        EntityExtractor instance
    """
    return EntityExtractor(
        language=language,
        model_name=model_name
    )
