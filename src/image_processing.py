"""
Image processing utilities for business card OCR.
Handles image loading, preprocessing, and enhancement.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Union
import logging
from pathlib import Path
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from the given path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: Loaded image in BGR format or None if loading fails
    """
    try:
        # Read image in BGR format (OpenCV default)
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.exception(f"Error loading image {image_path}: {str(e)}")
        return None

def preprocess_image(
    image: np.ndarray, 
    target_size: Tuple[int, int] = None,
    denoise: bool = True,
    enhance_contrast: bool = True,
    adaptive_threshold: bool = True
) -> np.ndarray:
    """
    Preprocess the image for better OCR results.
    
    Args:
        image: Input image in BGR format
        target_size: Target size as (width, height) or None to keep original
        denoise: Whether to apply denoising
        enhance_contrast: Whether to enhance contrast
        adaptive_threshold: Whether to use adaptive thresholding
        
    Returns:
        Preprocessed grayscale image
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize if target size is provided
        if target_size is not None:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Denoising
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Contrast enhancement using CLAHE
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Adaptive thresholding
        if adaptive_threshold:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        return gray
    except Exception as e:
        logger.exception(f"Error preprocessing image: {str(e)}")
        return image

def enhance_image_quality(
    image: np.ndarray,
    brightness: float = 1.2,
    contrast: float = 1.2,
    sharpness: float = 1.5
) -> np.ndarray:
    """
    Enhance image quality using PIL for better OCR results.
    
    Args:
        image: Input image (BGR format)
        brightness: Brightness enhancement factor
        contrast: Contrast enhancement factor
        sharpness: Sharpness enhancement factor
        
    Returns:
        Enhanced image (BGR format)
    """
    try:
        # Convert to PIL Image (RGB)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness)
            
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast)
            
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(sharpness)
        
        # Convert back to BGR for OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.exception(f"Error enhancing image: {str(e)}")
        return image

def detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect edges in the image using Canny edge detection.
    
    Args:
        image: Input grayscale image
        
    Returns:
        Edge-detected image
    """
    try:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Use Canny edge detection
        edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
        return edges
    except Exception as e:
        logger.exception(f"Error detecting edges: {str(e)}")
        return image

def find_document_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the largest rectangle contour in the image.
    
    Args:
        image: Input image (grayscale or edge-detected)
        
    Returns:
        Numpy array of contour points or None if not found
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
            
        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Approximate the contour to a polygon
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # If we found a quadrilateral
            if len(approx) == 4:
                return approx
                
        return None
    except Exception as e:
        logger.exception(f"Error finding document contour: {str(e)}")
        return None

def perspective_transform(
    image: np.ndarray, 
    contour: np.ndarray
) -> Optional[np.ndarray]:
    """
    Apply perspective transform to get a top-down view of the document.
    
    Args:
        image: Input image
        contour: 4-point contour of the document
        
    Returns:
        Warped image or None if transformation fails
    """
    try:
        # Get the rectangle vertices and order them
        rect = order_points(contour.reshape(4, 2))
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        max_width = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        max_height = max(int(heightA), int(heightB))
        
        # Define the destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    except Exception as e:
        logger.exception(f"Error in perspective transform: {str(e)}")
        return None

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in clockwise order starting from top-left.
    
    Args:
        pts: Array of 4 points
        
    Returns:
        Ordered points as (tl, tr, br, bl)
    """
    # Initialize the ordered points array
    rect = np.zeros((4, 2), dtype="float32")
    
    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    
    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    
    return rect

def process_business_card(
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    save_processed: bool = True
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process a business card image for OCR.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the processed image
        save_processed: Whether to save the processed image
        
    Returns:
        Tuple of (original_image, processed_image) or (None, None) on failure
    """
    try:
        # Load the image
        original = load_image(image_path)
        if original is None:
            return None, None
        
        # Enhance image quality
        enhanced = enhance_image_quality(original)
        
        # Convert to grayscale and preprocess
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        preprocessed = preprocess_image(gray)
        
        # Find edges and contours
        edges = detect_edges(preprocessed)
        contour = find_document_contour(edges)
        
        # Apply perspective transform if contour is found
        if contour is not None:
            warped = perspective_transform(enhanced, contour)
            if warped is not None:
                # Convert to grayscale for final processing
                processed = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                processed = preprocess_image(processed)
            else:
                # Fallback to enhanced grayscale if perspective transform fails
                processed = preprocess_image(gray)
        else:
            # Fallback to enhanced grayscale if no contour is found
            processed = preprocess_image(gray)
        
        # Save the processed image if requested
        if save_processed and output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), processed)
        
        return original, processed
    except Exception as e:
        logger.exception(f"Error processing business card {image_path}: {str(e)}")
        return None, None
