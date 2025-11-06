"""
Eto gumagana kaso di ata pwede gumamit ng
pytesseract sa final project so meh
"""
import cv2
import numpy as np
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# =============================================================================
# STEP 1: IMAGE PYRAMID
# =============================================================================
def pyramid(image, scale=1.5, minSize=(30, 30)):
    """Yields images at different scales (an image pyramid) using cv2.resize."""
    # yield the original image
    yield image
    
    while True:
        # compute the new width
        w = int(image.shape[1] / scale)
        
        # compute the new height, maintaining aspect ratio
        r = w / float(image.shape[1])
        h = int(image.shape[0] * r)
        
        # new dimensions
        dim = (w, h)
        
        # resize the image using cv2 (INTER_AREA is good for shrinking)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        # if the resized image is too small, stop
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
            
        # yield the next image in the pyramid
        yield image

# =============================================================================
# STEP 2: PREPROCESSING
# =============================================================================
def preprocess_image(image):
    """Converts to grayscale and applies a bilateral filter."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
    return bfilter

# =============================================================================
# STEP 3: PLATE LOCALIZATION
# =============================================================================
def find_plate_contour(processed_image):
    """Finds the 4-point contour of the license plate."""
    try:
        edged = cv2.Canny(processed_image, 30, 200) 
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        plate_contour = None
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

            if len(approx) == 4:
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Tune this aspect ratio range
                if 2.0 < aspect_ratio < 4.5 and w > 30 and h > 15:
                    plate_contour = approx
                    break
        
        return plate_contour
    except Exception as e:
        print(f"Error in find_plate_contour: {e}")
        return None

# =============================================================================
# STEP 4: PLATE ISOLATION (PERSPECTIVE TRANSFORM)
# =============================================================================
def crop_plate(image, plate_contour, plate_type="car"):
    """Crops and warps the 4-point contour to a flat, top-down image. Supports car and motorcycle plate sizes."""
    try:
        pts = plate_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left

        """
        For car plates, use 390x140
        For motorcycle plates, use 235x135
        """        
        if plate_type == "motorcycle":
            target_w = 235
            target_h = 135
        else:  # default to car
            target_w = 390
            target_h = 140

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (target_w, target_h))

        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        return warped_gray
    except Exception as e:
        print(f"Error in crop_plate: {e}")
        return None

# =============================================================================
# STEP 5: TEMPLATE LOADING
# =============================================================================
def load_templates(template_directory="templates"):
    """Loads all templates from a directory into a dictionary."""
    templates = {}
    if not os.path.exists(template_directory):
        print(f"Error: Template directory not found at {template_directory}")
        return None

    print(f"Loading templates from {template_directory}...")
    for filename in os.listdir(template_directory):
        if filename.endswith(('.png', '.jpg', '.bmp')):
            char_name = os.path.splitext(filename)[0]
            template_img = cv2.imread(os.path.join(template_directory, filename), 
                                      cv2.IMREAD_GRAYSCALE)
            
            if template_img is None: continue
            _, template_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
            templates[char_name] = template_thresh
            
    print(f"Loaded {len(templates)} templates.")
    return templates

# =============================================================================
# STEP 6: CHARACTER SEGMENTATION
# =============================================================================
def segment_characters(warped_plate):
    """Finds and sorts character contours from the straightened plate."""
    try:
        thresh = cv2.threshold(warped_plate, 0, 255, 
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        character_rois = []
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            
            # ** Tune these character heuristics **
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 1.0 and h > 20 and w > 5:
                character_rois.append((x, y, w, h))

        if not character_rois:
            return None, None

        character_rois = sorted(character_rois, key=lambda roi: roi[0])

        characters = []
        for (x, y, w, h) in character_rois:
            char_image = thresh[y:y+h, x:x+w]
            characters.append(char_image)
            
        return characters, character_rois, thresh
    
    except Exception as e:
        print(f"Error in segment_characters: {e}")
        return None, None, None

# =============================================================================
# STEP 7: CHARACTER RECOGNITION (TEMPLATE MATCHING)
# =============================================================================
def recognize_characters_pytesseract(warped_plate):
    """Recognizes characters using pytesseract OCR."""
    # Preprocess for pytesseract
    # Resize for better OCR accuracy
    scale_percent = 200
    width = int(warped_plate.shape[1] * scale_percent / 100)
    height = int(warped_plate.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(warped_plate, dim, interpolation=cv2.INTER_LINEAR)
    # Threshold to binary
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pytesseract expects RGB
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    # OCR config: only alphanumeric, single line
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(rgb, config=custom_config)
    # Clean up output
    plate_text = ''.join(filter(str.isalnum, text))
    return plate_text

# =============================================================================
# === MAIN PIPELINE ===
# =============================================================================
def main_pipeline(image_path, template_dir, show_steps=True):
    """Runs the complete ANPR pipeline using pytesseract for OCR."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Exiting: Could not load image from {image_path}")
        return None
    found_plate_contour = None
    original_copy = original_image.copy()

    print("Finding plate using image pyramid...")
    for (i, scaled_image) in enumerate(pyramid(original_copy, scale=1.5)):
        if scaled_image.shape[0] < 30 or scaled_image.shape[1] < 30:
            break
        processed = preprocess_image(scaled_image)
        contour = find_plate_contour(processed)
        if contour is not None:
            scale_factor = original_image.shape[1] / float(scaled_image.shape[1])
            found_plate_contour = (contour * scale_factor).astype(int)
            print("Plate contour found!")
            if show_steps:
                cv2.drawContours(original_image, [found_plate_contour], -1, (0, 255, 0), 3)
                cv2.imshow("1. Plate Contour on Original Image", original_image)
            break
    if found_plate_contour is None:
        print("Plate not found.")
        return None
    cropped_plate = crop_plate(original_copy, found_plate_contour)
    if cropped_plate is None:
        print("Could not crop plate.")
        return None
    if show_steps:
        cv2.imshow("2. Cropped Plate", cropped_plate)
    # Use pytesseract OCR directly on the cropped plate
    plate_text = recognize_characters_pytesseract(cropped_plate)
    print("---------------------------------")
    print(f"  Detected Plate Text: {plate_text}  ")
    print("---------------------------------")
    if show_steps:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return plate_text

# =============================================================================
# === RUN THE PROJECT ===
# =============================================================================
if __name__ == "__main__":
    # --- Configuration ---
    IMAGE_PATH = "plate_with_spaces.JPG" 
    TEMPLATE_DIR = "templates"
    SHOW_VISUAL_STEPS = True
    # ---------------------
    
    main_pipeline(IMAGE_PATH, TEMPLATE_DIR, show_steps=SHOW_VISUAL_STEPS)