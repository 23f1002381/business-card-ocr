"""
OCR Processing Module

This module provides a unified interface for text extraction from images
using multiple OCR engines with fallback mechanisms.
"""
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2

# Initialize logger
logger = logging.getLogger(__name__)

class OCRProcessor:
    """
    Handles OCR processing using multiple engines with fallback mechanisms.
    
    Primary Engine: EasyOCR
    Fallback Engine: Tesseract OCR
    """
    
    def __init__(self, languages: List[str] = ['en'], use_gpu: bool = False):
        """
        Initialize the OCR processor.
        
        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            use_gpu: Whether to use GPU if available
        """
        self.languages = languages
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        self.tesseract_available = self._check_tesseract_installed()
        
        # Initialize EasyOCR if available
        try:
            import easyocr
            self.easyocr_reader = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                model_storage_directory=str(Path.home() / '.easyocr/model'),
                download_enabled=True
            )
            logger.info("EasyOCR initialized successfully")
        except ImportError:
            logger.warning("EasyOCR not available. Falling back to Tesseract.")
        except Exception as e:
            logger.error(f"Error initializing EasyOCR: {str(e)}")
    
    def _check_tesseract_installed(self) -> bool:
        """Check if Tesseract OCR is installed and accessible."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except (ImportError, pytesseract.TesseractNotFoundError):
            logger.warning("Tesseract OCR not found. Install it for fallback support.")
            return False
        except Exception as e:
            logger.error(f"Error checking Tesseract installation: {str(e)}")
            return False
    
    def process_image(
        self,
        image: Union[str, Path, np.ndarray],
        engine: str = 'auto',
        **kwargs
    ) -> Tuple[bool, str, List[Dict]]:
        """
        Process an image and extract text using the specified OCR engine.
        
        Args:
            image: Input image (file path, Path object, or numpy array)
            engine: OCR engine to use ('easyocr', 'tesseract', or 'auto')
            **kwargs: Additional arguments for the OCR engine
            
        Returns:
            Tuple of (success, message, results)
            results is a list of dictionaries with keys: 'text', 'confidence', 'bounding_box'
        """
        try:
            # Convert image to numpy array if it's a file path
            if isinstance(image, (str, Path)):
                image_path = str(image)
                if not Path(image_path).exists():
                    return False, f"Image file not found: {image_path}", []
                
                try:
                    img_array = cv2.imread(image_path)
                    if img_array is None:
                        return False, f"Failed to load image: {image_path}", []
                except Exception as e:
                    return False, f"Error loading image: {str(e)}", []
            elif isinstance(image, np.ndarray):
                img_array = image.copy()
            else:
                return False, "Unsupported image format. Use file path or numpy array.", []
            
            # Convert to RGB if needed (for Tesseract)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            # Select OCR engine
            if engine == 'auto':
                if self.easyocr_reader is not None:
                    return self._process_with_easyocr(img_array, **kwargs)
                elif self.tesseract_available:
                    return self._process_with_tesseract(img_rgb, **kwargs)
                else:
                    return False, "No OCR engine available. Please install EasyOCR or Tesseract.", []
            
            elif engine == 'easyocr' and self.easyocr_reader is not None:
                return self._process_with_easyocr(img_array, **kwargs)
            
            elif engine == 'tesseract' and self.tesseract_available:
                return self._process_with_tesseract(img_rgb, **kwargs)
            
            else:
                return False, f"Requested OCR engine '{engine}' is not available.", []
                
        except Exception as e:
            error_msg = f"Error in OCR processing: {str(e)}"
            logger.exception(error_msg)
            return False, error_msg, []
    
    def _process_with_easyocr(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[bool, str, List[Dict]]:
        """Process image using EasyOCR."""
        try:
            # Convert to RGB if needed (EasyOCR expects RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run OCR
            results = self.easyocr_reader.readtext(
                image_rgb,
                batch_size=1,
                detail=1,
                paragraph=False,
                **kwargs
            )
            
            # Format results
            formatted_results = []
            for (bbox, text, confidence) in results:
                formatted_results.append({
                    'text': text.strip(),
                    'confidence': float(confidence),
                    'bounding_box': bbox,
                    'engine': 'easyocr'
                })
            
            return True, "Success", formatted_results
            
        except Exception as e:
            error_msg = f"EasyOCR processing failed: {str(e)}"
            logger.exception(error_msg)
            return False, error_msg, []
    
    def _process_with_tesseract(
        self,
        image: np.ndarray,
        **kwargs
    ) -> Tuple[bool, str, List[Dict]]:
        """Process image using Tesseract OCR."""
        try:
            import pytesseract
            from pytesseract import Output
            
            # Configure Tesseract parameters
            config = {
                'output_type': Output.DICT,
                'lang': '+'.join(self.languages) if self.languages else 'eng',
                'config': '--psm 6 --oem 3'  # PSM 6: Assume a single uniform block of text
            }
            
            # Update with any custom config
            config.update(kwargs.get('tesseract_config', {}))
            
            # Run Tesseract OCR
            results = pytesseract.image_to_data(image, **config)
            
            # Process results
            formatted_results = []
            n_boxes = len(results['level'])
            
            for i in range(n_boxes):
                if int(results['conf'][i]) > 0:  # Only include results with confidence > 0
                    x, y, w, h = (
                        results['left'][i],
                        results['top'][i],
                        results['width'][i],
                        results['height'][i]
                    )
                    
                    # Calculate bounding box coordinates (x1, y1, x2, y2, x3, y3, x4, y4)
                    bbox = [
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ]
                    
                    formatted_results.append({
                        'text': results['text'][i].strip(),
                        'confidence': float(results['conf'][i]) / 100.0,  # Convert to 0-1 range
                        'bounding_box': bbox,
                        'engine': 'tesseract'
                    })
            
            return True, "Success", formatted_results
            
        except Exception as e:
            error_msg = f"Tesseract processing failed: {str(e)}"
            logger.exception(error_msg)
            return False, error_msg, []
    
    def batch_process(
        self,
        image_paths: List[Union[str, Path]],
        engine: str = 'auto',
        **kwargs
    ) -> Dict[str, Tuple[bool, str, List[Dict]]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths or numpy arrays
            engine: OCR engine to use
            **kwargs: Additional arguments for the OCR engine
            
        Returns:
            Dictionary mapping image paths to (success, message, results) tuples
        """
        results = {}
        
        for img_path in image_paths:
            if isinstance(img_path, (str, Path)):
                img_key = str(img_path)
            else:
                img_key = f"image_{len(results)}"
                
            success, message, ocr_results = self.process_image(img_path, engine, **kwargs)
            results[img_key] = (success, message, ocr_results)
        
        return results


def create_ocr_processor(
    languages: List[str] = None,
    use_gpu: bool = False
) -> Optional[OCRProcessor]:
    """
    Factory function to create an OCR processor instance.
    
    Args:
        languages: List of language codes (e.g., ['en', 'fr'])
        use_gpu: Whether to use GPU if available
        
    Returns:
        OCRProcessor instance or None if no OCR engine is available
    """
    if languages is None:
        languages = ['en']
    
    try:
        return OCRProcessor(languages=languages, use_gpu=use_gpu)
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {str(e)}")
        return None
