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

### Python Implementation
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

### JavaScript/TypeScript Implementation
```typescript
import Tesseract from 'tesseract.js';
import Jimp from 'jimp';
import { ImageAnnotatorClient } from '@google-cloud/vision';

// Tesseract.js OCR
async function tesseractOCR(imagePath: string): Promise<string> {
  const { data: { text } } = await Tesseract.recognize(imagePath, 'eng', {
    logger: m => console.log(m)
  });
  return text;
}

// Image preprocessing with Jimp
async function preprocessImage(imagePath: string): Promise<Buffer> {
  const image = await Jimp.read(imagePath);
  
  const processedImage = image
    .greyscale()                    // Convert to grayscale
    .contrast(0.5)                  // Increase contrast
    .normalize()                    // Normalize the image
    .threshold({ max: 128 });       // Apply threshold
  
  return processedImage.getBufferAsync(Jimp.MIME_PNG);
}

// Google Vision API OCR
async function googleVisionOCR(imagePath: string): Promise<string> {
  const client = new ImageAnnotatorClient();
  
  const [result] = await client.textDetection(imagePath);
  const detections = result.textAnnotations;
  
  return detections && detections[0] ? detections[0].description || '' : '';
}

// OCR with confidence scoring
async function ocrWithConfidence(imagePath: string) {
  const worker = await Tesseract.createWorker();
  await worker.loadLanguage('eng');
  await worker.initialize('eng');
  
  const { data } = await worker.recognize(imagePath);
  
  const results = {
    text: data.text,
    confidence: data.confidence,
    words: data.words.map(word => ({
      text: word.text,
      confidence: word.confidence,
      bbox: word.bbox
    }))
  };
  
  await worker.terminate();
  return results;
}

// Browser-based OCR
function browserOCR(imageFile: File): Promise<string> {
  return new Promise((resolve, reject) => {
    Tesseract.recognize(
      imageFile,
      'eng',
      {
        logger: m => console.log(m)
      }
    ).then(({ data: { text } }) => {
      resolve(text);
    }).catch(reject);
  });
}
```

### Advanced Document Processing (JavaScript/TypeScript)
```typescript
import sharp from 'sharp';
import { PDFExtract } from 'pdf.js-extract';

// Advanced image preprocessing with Sharp
async function advancedPreprocessing(imagePath: string): Promise<Buffer> {
  return await sharp(imagePath)
    .greyscale()
    .normalize()
    .linear(1.2, 0)        // Adjust brightness/contrast
    .sharpen()             // Sharpen the image
    .png()
    .toBuffer();
}

// PDF text extraction
async function extractTextFromPDF(pdfPath: string): Promise<string> {
  const pdfExtract = new PDFExtract();
  const options = {}; // Optional configuration
  
  return new Promise((resolve, reject) => {
    pdfExtract.extract(pdfPath, options, (err, data) => {
      if (err) reject(err);
      
      const text = data.pages
        .map(page => page.content
          .map(item => item.str)
          .join(' ')
        )
        .join('\n');
      
      resolve(text);
    });
  });
}

// Document processing pipeline
class DocumentProcessor {
  async processDocument(file: File | string): Promise<{
    text: string;
    confidence: number;
    metadata: any;
  }> {
    let processedImage: Buffer;
    
    if (typeof file === 'string') {
      processedImage = await advancedPreprocessing(file);
    } else {
      // Handle File object for browser usage
      const arrayBuffer = await file.arrayBuffer();
      processedImage = await sharp(Buffer.from(arrayBuffer))
        .greyscale()
        .normalize()
        .png()
        .toBuffer();
    }
    
    const ocrResult = await ocrWithConfidence(processedImage);
    
    return {
      text: ocrResult.text,
      confidence: ocrResult.confidence,
      metadata: {
        wordCount: ocrResult.words.length,
        processingDate: new Date().toISOString()
      }
    };
  }
}
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
