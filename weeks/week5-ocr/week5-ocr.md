# Week 5: Optical Character Recognition (OCR)

## Learning Objectives
- Understand OCR fundamentals and applications
- Learn to use modern OCR libraries and cloud services
- Implement document processing pipelines
- Extract structured data from various document types

## Topics Covered

### 1. OCR Fundamentals
- History and evolution of OCR
- Traditional vs Modern OCR approaches
- OCR challenges (fonts, layouts, quality)
- Performance metrics and evaluation

### 2. OCR Libraries and Tools
- **Tesseract**: Installation, configuration, languages
- **EasyOCR**: Multi-language support, ease of use
- **PaddleOCR**: High accuracy, lightweight
- **Cloud OCR APIs**: Google Vision, AWS Textract, Azure Computer Vision

### 3. Image Preprocessing
- Image quality enhancement
- Noise reduction and filtering
- Skew correction and alignment
- Binarization and thresholding
- Layout analysis and text detection

### 4. Document Types and Specialized OCR
- Printed text documents
- Handwritten text recognition
- Forms and structured documents
- Tables and invoice processing
- Multi-language documents

### 5. Post-processing and Accuracy Improvement
- Spell checking and correction
- Context-aware error correction
- Language models for OCR
- Confidence scoring and validation

### 6. Integration with AI Systems
- OCR + LLM pipelines
- Structured data extraction
- Document understanding workflows
- Real-time OCR applications

## Exercises

1. **Basic OCR Implementation**
   - Set up Tesseract and EasyOCR
   - Process various document types
   - Compare accuracy across different tools
   - Implement preprocessing pipeline

2. **Document Processing System**
   - Build an invoice data extraction system
   - Handle different document layouts
   - Implement confidence scoring
   - Add error handling and validation

3. **Advanced OCR Pipeline**
   - Combine OCR with LLM for understanding
   - Implement table extraction
   - Build a document classification system
   - Create a web interface for document upload

## Code Examples

```python
import cv2
import pytesseract
from PIL import Image
import easyocr

# Tesseract OCR
def tesseract_ocr(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# EasyOCR
def easy_ocr(image_path):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    text = ' '.join([result[1] for result in results])
    return text

# Image preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise removal
    denoised = cv2.medianBlur(gray, 5)
    
    # Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, 
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

# Cloud OCR example (Google Vision API)
from google.cloud import vision

def google_vision_ocr(image_path):
    client = vision.ImageAnnotatorClient()
    
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    return texts[0].description if texts else ""
```

## Resources

### OCR Libraries (Python)
- [Tesseract Documentation](https://tesseract-ocr.github.io/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [PyTesseract Documentation](https://pypi.org/project/pytesseract/)

### JavaScript/TypeScript OCR
- [Tesseract.js](https://github.com/naptha/tesseract.js) - OCR for JavaScript/Node.js
- [OCR-Space API](https://ocr.space/ocrapi) - Cloud OCR with JavaScript SDK
- [Google Cloud Vision Node.js](https://cloud.google.com/vision/docs/libraries#client-libraries-install-nodejs)
- [Azure Computer Vision SDK](https://docs.microsoft.com/en-us/javascript/api/@azure/cognitiveservices-computervision/)
- [AWS Textract JavaScript SDK](https://docs.aws.amazon.com/AWSJavaScriptSDK/latest/AWS/Textract.html)

### Image Processing (JavaScript/TypeScript)
- [Jimp](https://github.com/oliver-moran/jimp) - Image processing library for Node.js
- [Sharp](https://github.com/lovell/sharp) - High-performance image processing
- [Canvas API](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) - Browser image manipulation
- [Fabric.js](http://fabricjs.com/) - Canvas library for interactive graphics

### Cloud OCR APIs
- [Google Cloud Vision API](https://cloud.google.com/vision/docs/ocr)
- [AWS Textract](https://aws.amazon.com/textract/)
- [Azure Computer Vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/)
- [Microsoft Form Recognizer](https://azure.microsoft.com/en-us/services/form-recognizer/)

### Additional Resources
- [OpenCV Image Processing Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- Academic papers on document analysis and recognition
- [Document AI Datasets](https://paperswithcode.com/datasets?task=optical-character-recognition)

## Next Week Preview
Week 6 will focus on AI Agents and autonomous systems that can perform complex tasks.
