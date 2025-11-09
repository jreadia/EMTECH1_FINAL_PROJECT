import cv2
import numpy as np
import os

# =============================================================================
# STEP 1: IMAGE PYRAMID
# =============================================================================
def pyramid(image, scale=1.5, min_size=(20, 20)):
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
        else:  # default to car:
            target_w = 390
            target_h = 140

        dst = np.array([
            [0, 0],
            [target_w - 1, 0],
            [target_w - 1, target_h - 1],
            [0, target_h - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(image, M, (target_w, target_h))

        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        return cropped_gray
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
            # Use THRESH_BINARY to match the segmented characters (white on black)
            _, template_thresh = cv2.threshold(template_img, 127, 255, cv2.THRESH_BINARY)
            
            # CRITICAL: Tightly crop the template to remove padding/whitespace
            coords = cv2.findNonZero(template_thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                template_thresh = template_thresh[y:y+h, x:x+w]
            
            templates[char_name] = template_thresh
            
            # DEBUG: Save templates to see polarity
            # if char_name in ['0', 'N', 'F', '6']:
            #     cv2.imwrite(f"debug_template_{char_name}_thresh.jpg", template_thresh)
            #     print(f"DEBUG: Saved template '{char_name}' - thresholded and cropped")
            
    print(f"Loaded {len(templates)} templates.")
    print(f"Available characters: {sorted(templates.keys())}")
    return templates

# =============================================================================
# STEP 6: CHARACTER SEGMENTATION
# =============================================================================
def segment_characters(warped_plate):
    """Finds, deskews, and sorts character contours."""
    try:
        # Same thresholding as before
        thresh = cv2.threshold(warped_plate, 0, 255, 
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        character_rois = [] # (x, y, w, h) for sorting
        character_images = [] # The deskewed images
        
        for c in contours:
            # Get the upright bounding box for filtering
            (x, y, w, h) = cv2.boundingRect(c)
            
            # ** Tune these character heuristics **
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 1.0 and h > 20 and w > 5:
                
                # --- THIS IS THE NEW PART ---
                # 1. Get the minimum area rectangle
                rect = cv2.minAreaRect(c)
                (center, (width, height), angle) = rect

                # 2. Get the 4 corners of the rotated box
                box = cv2.boxPoints(rect)
                box = np.int64(box)

                # 3. Order the 4 points
                # (This is a simplified version of our straighten_plate logic)
                pts = box.reshape(4, 2)
                ordered_pts = np.zeros((4, 2), dtype="float32")

                s = pts.sum(axis=1)
                ordered_pts[0] = pts[np.argmin(s)] # Top-left
                ordered_pts[2] = pts[np.argmax(s)] # Bottom-right

                diff = np.diff(pts, axis=1)
                ordered_pts[1] = pts[np.argmin(diff)] # Top-right
                ordered_pts[3] = pts[np.argmax(diff)] # Bottom-left
                
                # 4. Define the target (straight) dimensions
                # We want a standard-sized, upright character
                
                # Ensure width is the larger dimension if the char is on its side
                if width < height:
                    width, height = height, width
                    
                # ** Tune this standard size (e.g., 20w x 40h) **
                target_w = 20
                target_h = 40
                
                dst = np.array([
                    [0, 0],
                    [target_w - 1, 0],
                    [target_w - 1, target_h - 1],
                    [0, target_h - 1]], dtype="float32")

                # 5. Get the transform matrix and warp
                M = cv2.getPerspectiveTransform(ordered_pts, dst)
                # We warp from the *thresholded* plate
                deskewed_char = cv2.warpPerspective(thresh, M, (target_w, target_h))
                
                # Add the character image
                character_images.append(deskewed_char)
                # Add the x-coord for sorting
                character_rois.append((x, y, w, h))

        if not character_rois:
            return None, None, None

        # Sort the characters based on their *original* x-coordinate
        # This zips them together, sorts by x, then unzips
        zipped = sorted(zip(character_rois, character_images), key=lambda item: item[0][0])
        sorted_rois, sorted_characters = zip(*zipped)

        return sorted_characters, sorted_rois, thresh
    
    except Exception as e:
        print(f"Error in segment_characters: {e}")
        return None, None, None

# =============================================================================
# STEP 7: CHARACTER RECOGNITION (TEMPLATE MATCHING)
# =============================================================================
def recognize_characters_template_matching(character_images, templates):
    """Identifies characters using cv2.matchTemplate."""
    if not templates: return "Template DB not loaded"
    if not character_images: return ""

    plate_text = ""
    
    for idx, char_img in enumerate(character_images):
        best_match_score = -1
        best_match_char = "?"
        
        try:
            # ** Tune this standard height **
            STANDARD_HEIGHT = 40.0
            h, w = char_img.shape
            scale = STANDARD_HEIGHT / h
            char_resized = cv2.resize(char_img, (int(w * scale), int(STANDARD_HEIGHT)), 
                                      interpolation=cv2.INTER_AREA)
            print(f"  Char {idx+1}: Original size={char_img.shape}, Resized={char_resized.shape}")
        except cv2.error as e:
            print(f"  Char {idx+1}: Resize error: {e}")
            continue

        templates_tested = 0
        templates_tested = 0
        for char_name, template in templates.items():
            try:
                t_h, t_w = template.shape
                scale = char_resized.shape[0] / float(t_h)
                template_resized = cv2.resize(template, (int(t_w * scale), char_resized.shape[0]), 
                                              interpolation=cv2.INTER_AREA)
            except cv2.error:
                continue

            # Skip only if template is WAY too wide (more than 2x the character width)
            if template_resized.shape[1] > char_resized.shape[1] * 2.0:
                continue
            
            templates_tested += 1
            
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
        
        # DEBUG: Print matching information
        # print(f"  Char {idx+1}: Tested {templates_tested} templates, Best match = '{best_match_char}' with score = {best_match_score:.3f}")
                
        # ** Tune this confidence threshold **
        # Lowered to 0.2 for better recognition with challenging characters
        if best_match_score > 0.2: 
            plate_text += best_match_char
        else:
            plate_text += "?"

    return plate_text

# =============================================================================
# STEP 2.5: HOG-BASED PLATE DETECTION
# =============================================================================
def sliding_window(image, step_size, window_size):
    """Yields windows at different positions in the image."""
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_plate_hog(image, min_score=0.15):
    """
    Detects license plate regions using HOG features and sliding window.
    Returns bounding boxes of potential plate regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply bilateral filter for noise reduction
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection to focus on regions with high gradient
    edges = cv2.Canny(filtered, 30, 200)
    
    # Initialize HOG descriptor
    win_size = (128, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    potential_plates = []
    
    # Different window sizes to detect plates at various scales
    window_sizes = [(120, 40), (160, 50), (200, 60), (240, 80)]
    step_size = 20
    
    for window_size in window_sizes:
        if window_size[0] > image.shape[1] or window_size[1] > image.shape[0]:
            continue
            
        for (x, y, window) in sliding_window(edges, step_size, window_size):
            if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                continue
            
            # Calculate edge density in the window
            edge_density = np.sum(window > 0) / (window_size[0] * window_size[1])
            
            # Compute HOG features using cv2.HOGDescriptor
            try:
                # Resize window to HOG descriptor size
                resized_window = cv2.resize(window, win_size, interpolation=cv2.INTER_AREA)
                hog_features = hog.compute(resized_window)
                
                if hog_features is None:
                    continue
                
                # Calculate feature magnitude as a simple score
                feature_magnitude = np.mean(np.abs(hog_features))
                
                # Combined score: edge density + HOG feature strength
                score = edge_density * 0.7 + (feature_magnitude / 100) * 0.3
                
            except Exception as e:
                continue
            
            # Check aspect ratio (plates are wider than tall)
            aspect_ratio = window_size[0] / window_size[1]
            if 2.0 < aspect_ratio < 4.5 and score > min_score:
                potential_plates.append((x, y, window_size[0], window_size[1], score))
    
    # Non-maximum suppression to remove overlapping detections
    if potential_plates:
        potential_plates = non_max_suppression(np.array(potential_plates), 0.3)
    
    return potential_plates

def non_max_suppression(boxes, overlap_threshold=0.3):
    """Removes overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    
    # Sort by confidence score (last column)
    boxes = boxes[boxes[:, 4].argsort()[::-1]]
    
    keep = []
    while len(boxes) > 0:
        keep.append(boxes[0])
        
        if len(boxes) == 1:
            break
            
        # Compute IoU with remaining boxes
        xx1 = np.maximum(boxes[0, 0], boxes[1:, 0])
        yy1 = np.maximum(boxes[0, 1], boxes[1:, 1])
        xx2 = np.minimum(boxes[0, 0] + boxes[0, 2], boxes[1:, 0] + boxes[1:, 2])
        yy2 = np.minimum(boxes[0, 1] + boxes[0, 3], boxes[1:, 1] + boxes[1:, 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        overlap = (w * h) / (boxes[1:, 2] * boxes[1:, 3])
        
        boxes = boxes[1:][overlap <= overlap_threshold]
    
    return np.array(keep)

def hog_to_contour(x, y, w, h):
    """Converts HOG bounding box to contour format."""
    return np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.int32)

# =============================================================================
# === MAIN PIPELINE ===
# =============================================================================
def main_pipeline(image_path, template_dir, show_steps=True, use_hog=True):
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

    # Try HOG detection first if enabled
    if use_hog:
        print("Finding plate using HOG detection...")
        hog_plates = detect_plate_hog(original_copy)
        
        if hog_plates is not None and len(hog_plates) > 0:
            # Use the highest scoring detection
            x, y, w, h, score = hog_plates[0]
            found_plate_contour = hog_to_contour(int(x), int(y), int(w), int(h))
            print(f"Plate found using HOG (score: {score:.3f})!")
            
            if show_steps:
                hog_visual = original_image.copy()
                cv2.rectangle(hog_visual, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
                cv2.putText(hog_visual, f"HOG: {score:.2f}", (int(x), int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow("0. HOG Detection", hog_visual)
    
    # Fallback to contour detection if HOG didn't find anything
    if found_plate_contour is None:
        print("Finding plate using image pyramid and contours...")
        for (i, scaled_image) in enumerate(pyramid(original_copy, scale=1.5)):
            if scaled_image.shape[0] < 30 or scaled_image.shape[1] < 30:
                break
                
            processed = preprocess_image(scaled_image)
            contour = find_plate_contour(processed)
            
            if contour is not None:
                scale_factor = original_image.shape[1] / float(scaled_image.shape[1])
                found_plate_contour = (contour * scale_factor).astype(int)
                print("Plate contour found using edge detection!")
                break
            
    if found_plate_contour is None:
        print("Plate not found.")
        return None
    
    if show_steps:
        cv2.drawContours(original_image, [found_plate_contour], -1, (0, 255, 0), 3)
        cv2.imshow("1. Plate Contour on Original Image", original_image)

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
        
        # DEBUG: Save first few characters to see them clearly
        # for i in range(min(4, len(characters))):
        #     cv2.imwrite(f"debug_char_{i}.jpg", characters[i])
        # print(f"DEBUG: Saved first {min(4, len(characters))} segmented characters")
        
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
    TEMPLATE_DIR = "templates"
    SHOW_VISUAL_STEPS = True
    USE_HOG_DETECTION = True  # Set to False to use only contour detection
    # ---------------------
    
    main_pipeline(IMAGE_PATH, TEMPLATE_DIR, show_steps=SHOW_VISUAL_STEPS, use_hog=USE_HOG_DETECTION)