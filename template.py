import cv2
import numpy as np
import os
# Note: 'import imutils' has been removed

# =============================================================================
# STEP 1: IMAGE PYRAMID (cv2-only)
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
                
                # ** Tune this aspect ratio range **
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
def straighten_plate(image, plate_contour):
    """Warps the 4-point contour to a flat, top-down image."""
    try:
        pts = plate_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        
        # Using a standard PH plate size (approx 2.78 ratio)
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
        print(f"Error in straighten_plate: {e}")
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
                box = np.int0(box)

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
    
    for char_img in character_images:
        best_match_score = -1
        best_match_char = "?"
        
        try:
            # ** Tune this standard height **
            STANDARD_HEIGHT = 40.0
            h, w = char_img.shape
            scale = STANDARD_HEIGHT / h
            char_resized = cv2.resize(char_img, (int(w * scale), int(STANDARD_HEIGHT)), 
                                      interpolation=cv2.INTER_AREA)
        except cv2.error:
            continue

        for char_name, template in templates.items():
            try:
                t_h, t_w = template.shape
                scale = char_resized.shape[0] / float(t_h)
                template_resized = cv2.resize(template, (int(t_w * scale), char_resized.shape[0]), 
                                              interpolation=cv2.INTER_AREA)
            except cv2.error:
                continue

            if template_resized.shape[1] > char_resized.shape[1]:
                continue
            
            result = cv2.matchTemplate(char_resized, template_resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > best_match_score:
                best_match_score = max_val
                best_match_char = char_name
                
        # ** Tune this confidence threshold **
        if best_match_score > 0.6: 
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
        
    warped_plate = straighten_plate(original_copy, found_plate_contour)
    if warped_plate is None:
        print("Could not straighten plate.")
        return None
        
    if show_steps:
        cv2.imshow("2. Straightened (Warped) Plate", warped_plate)
        
    characters, rois, binary_plate = segment_characters(warped_plate)
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
    IMAGE_PATH = "test_car.jpg" 
    TEMPLATE_DIR = "templates"
    SHOW_VISUAL_STEPS = True
    # ---------------------
    
    main_pipeline(IMAGE_PATH, TEMPLATE_DIR, show_steps=SHOW_VISUAL_STEPS)