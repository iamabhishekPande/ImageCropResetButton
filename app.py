## following code working fine but it is creating folder to save the cropped image
import cv2
import numpy as np
from PIL import Image
import os
from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)

class ImageProcessor:
    def __init__(self):
        # Initialize the crack detection pipeline
        self.crack_detection_pipeline = pipeline("image-classification", model="Taki3d/CrackDetection")
        # Set the coordinate folder path
        self.coordinate_folder_path = r'C:\Users\abhishek\Desktop\216Cordinates'

    def process_image(self, image_path):
        # Load the image using OpenCV
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(original_image)

        # Get a list of all text files in the coordinate folder
        coordinate_files = [file for file in os.listdir(self.coordinate_folder_path) if file.endswith(".txt")]

        # Initialize a variable to store all outlines
        all_outlines = []

        # Iterate through each coordinate file
        for file_name in coordinate_files:
            file_path = os.path.join(self.coordinate_folder_path, file_name)
            
            # Read coordinates from file
            with open(file_path, 'r') as file:
                polygon_points = [tuple(map(float, line.strip().split(','))) for line in file]

            # Convert the polygon points to numpy array
            roi_corners = np.array([polygon_points], dtype=np.int32)

            # Create a mask using the polygon
            mask = np.zeros_like(np.array(image), dtype=np.uint8)
            cv2.fillPoly(mask, roi_corners, (255, 255, 255))

            # Apply the mask to the original image
            masked_image = cv2.bitwise_and(original_image, mask)

            # Find contours in the mask
            contours, _ = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Initialize a list to store outlines
            outlines = []

            # Iterate through each contour
            for contour in contours:
                # Find the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the region of interest from the original image
                cropped_image = masked_image[y:y+h, x:x+w]

                # Convert the cropped image to a PIL Image
                cropped_image_pil = Image.fromarray(cropped_image)

                # Save the cropped image with the name of the coordinate file
                output_folder = os.path.join(os.path.dirname(image_path), "cropped_images")
                os.makedirs(output_folder, exist_ok=True)
                output_filename = f"{os.path.splitext(os.path.basename(file_name))[0]}_cropped.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cropped_image_pil.save(output_path)
                print(f"Cropped image saved to: {output_path}")

                # Apply crack detection on the cropped image
                crack_detection_result = self.crack_detection_pipeline(output_path)
                print("Crack detection result:", crack_detection_result)

                # Check if crack score > 0.30
                for result in crack_detection_result:
                    if result['label'] == 'crack' and result['score'] > 0.:
                        outlines.append((contour, (255, 0, 0), os.path.splitext(os.path.basename(file_name))[0]))  # Red outline for high crack score
                        break  # Once we find the score for 'crack', we break the loop
            
            all_outlines.extend(outlines)

        return all_outlines

    def draw_outlines(self, image_path, outlines):
        # Load the image using OpenCV
        original_image = cv2.imread(image_path)
        original_image_with_outlines = original_image.copy()  # Create a copy for drawing outlines

        # Draw outlines on the original image
        for contour, color, coord_file_name in outlines:
            cv2.drawContours(original_image_with_outlines, [contour], -1, color, 2)

            # Add coordinate file name as text annotation
            cv2.putText(original_image_with_outlines, coord_file_name,
                        (int(contour[:, :, 0].mean()), int(contour[:, :, 1].mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        return original_image_with_outlines

image_processor = ImageProcessor()

@app.route('/process_image', methods=['POST'])
def process_image_route():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'})

    image_file = request.files['image']

    # If the user does not select a file, the browser submits an empty file without a filename
    if image_file.filename == '':
        return jsonify({'error': 'No selected image'})

    # Save the image to a temporary location
    image_path = '/tmp/' + image_file.filename
    image_file.save(image_path)

    # Process the image
    outlines = image_processor.process_image(image_path)

    # Draw outlines on the original image
    original_image_with_outlines = image_processor.draw_outlines(image_path, outlines)

    # Save the image with outlines
    outlines_output_folder = os.path.join(os.path.dirname(image_path), "outlines")
    os.makedirs(outlines_output_folder, exist_ok=True)
    outlines_output_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_with_outlines.jpg"
    outlines_output_path = os.path.join(outlines_output_folder, outlines_output_filename)
    cv2.imwrite(outlines_output_path, cv2.cvtColor(original_image_with_outlines, cv2.COLOR_RGB2BGR))
    print(f"Image with outlines saved to: {outlines_output_path}")

    return "Processing completed"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
