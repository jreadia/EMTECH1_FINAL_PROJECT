import cv2
import numpy as np
import os

# =============================================================================
# STEP 1: IMAGE PYRAMID
# =============================================================================
def pyramid(image, scale=1.5, min_size=(30, 30)):
    """
    Yields images at different scales (an image pyramid).
    Each step reduces the image size by the given scale factor.
    Stops when the image is smaller than min_size.
    """
    yield image  # Yield the original image

    while True:
        # Calculate new dimensions
        new_width = int(image.shape[1] / scale)
        new_height = int(image.shape[0] / scale)

        # Stop if the image is too small
        if new_width < min_size[0] or new_height < min_size[1]:
            break

        # Resize and yield the scaled image
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
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
                if 2.0 < aspect_ratio < 3.5 and w > 30 and h > 15:
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
    """Crops and warps the 4-point contour to a flat, top-down image."""
    try:
        # Order the contour points using OpenCV's built-in function
        pts = plate_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Top-left
        rect[2] = pts[np.argmax(s)]      # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Top-right
        rect[3] = pts[np.argmax(diff)]   # Bottom-left

        # Set output size based on plate type
        target_sizes = {"car": (390, 140), "motorcycle": (235, 135)}
        target_w, target_h = target_sizes.get(plate_type, (390, 140))

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(image, M, (target_w, target_h))
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
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
    # Loop through each subdirectory (A-Z, 0-9)
    for char_folder in os.listdir(template_directory):
        folder_path = os.path.join(template_directory, char_folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.bmp')):
                template_img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
                if template_img is None: continue
                # Use THRESH_BINARY to match the segmented characters (white on black)
                _, template_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
                # CRITICAL: Tightly crop the template to remove padding/whitespace
                coords = cv2.findNonZero(template_thresh)
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    template_thresh = template_thresh[y:y+h, x:x+w]
                # Store as a list in case multiple templates per character
                if char_folder not in templates:
                    templates[char_folder] = []
                templates[char_folder].append(template_thresh)
    print(f"Loaded templates for {len(templates)} characters.")
    print(f"Available characters: {sorted(templates.keys())}")
    return templates

# =============================================================================
# STEP 6: CHARACTER SEGMENTATION
# =============================================================================
def segment_characters(warped_plate):
    """Segments characters using connected components (simpler than contours)."""
    try:
        thresh = cv2.threshold(warped_plate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
        character_images = []
        character_rois = []
        for i in range(1, num_labels):  # skip background
            x, y, w, h, _ = stats[i]
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 1.0 and h > 20 and w > 5:
                char_img = cv2.resize(thresh[y:y+h, x:x+w], (20, 40), interpolation=cv2.INTER_AREA)
                character_images.append(char_img)
                character_rois.append((x, y, w, h))
        if not character_rois:
            return None, None, None
        sorted_rois, sorted_characters = zip(*sorted(zip(character_rois, character_images), key=lambda item: item[0][0]))
        return sorted_characters, sorted_rois, thresh
    except Exception as e:
        print(f"Error in segment_characters: {e}")
        return None, None, None

# =============================================================================
# STEP 7: CHARACTER RECOGNITION (TEMPLATE MATCHING)
# =============================================================================
def recognize_characters_template_matching(character_images, templates):
    """Identifies characters using cv2.matchTemplate."""
    if not templates: 
        return "Template DB not loaded"
    if not character_images: 
        return ""

    plate_text = ""
    
    for idx, char_img in enumerate(character_images):
        best_match_score = -1
        best_match_char = "?"
        
        try:
            # ** Tune this standard height **
            STANDARD_HEIGHT = 50.0
            h, w = char_img.shape
            scale = STANDARD_HEIGHT / h
            char_resized = cv2.resize(char_img, (int(w * scale), int(STANDARD_HEIGHT)), 
                                      interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            print(f"  Char {idx+1}: Resize error: {e}")
            continue

        # Loop through all templates for each character
        for char_name, template_list in templates.items():
            for template in template_list:
                try:
                    t_h, t_w = template.shape
                    scale = char_resized.shape[0] / float(t_h)
                    template_resized = cv2.resize(template, (int(t_w * scale), char_resized.shape[0]), 
                                                  interpolation=cv2.INTER_AREA)
                except cv2.error:
                    continue
                
                # If template is wider than character, pad the character
                if template_resized.shape[1] > char_resized.shape[1]:
                    pad_width = template_resized.shape[1] - char_resized.shape[1]
                    char_padded = cv2.copyMakeBorder(char_resized, 0, 0, 0, pad_width, 
                                                     cv2.BORDER_CONSTANT, value=0)
                    result = cv2.matchTemplate(char_padded, template_resized, cv2.TM_CCOEFF_NORMED)
                else:
                    result = cv2.matchTemplate(char_resized, template_resized, cv2.TM_CCOEFF_NORMED)
                
                _, max_val, _, _ = cv2.minMaxLoc(result)

                if max_val > best_match_score:
                    best_match_score = max_val
                    best_match_char = char_name
    
        if best_match_score > 0.3: 
            plate_text += best_match_char
        else:
            plate_text += "?"

    return plate_text

# =============================================================================
# === MAIN PIPELINE ===
# =============================================================================
def main_pipeline(image_path, template_dir, show_steps=True):
    """Runs the complete ANPR pipeline."""
    
    template_db = load_templates(template_dir)
    if not template_db:
        print("Exiting: Template database is empty or could not be loaded.")
        return None
        
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

    characters, rois, binary_plate = segment_characters(cropped_plate)
    if characters is None:
        print("Could not segment characters from plate.")
        return None
        
    if show_steps:
        binary_with_boxes = cv2.cvtColor(binary_plate, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in rois:
            cv2.rectangle(binary_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("3. Segmented Characters", binary_with_boxes)
        
    plate_text = recognize_characters_template_matching(characters, template_db)
    
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
    IMAGE_PATH = "plate_with_spaces.jpg" 
    TEMPLATE_DIR = "template4"
    SHOW_VISUAL_STEPS = True
    # ---------------------
    
    main_pipeline(IMAGE_PATH, TEMPLATE_DIR, show_steps=SHOW_VISUAL_STEPS)