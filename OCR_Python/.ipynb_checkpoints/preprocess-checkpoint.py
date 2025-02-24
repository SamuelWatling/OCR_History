#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:35:53 2023

@author: michaelgiordano
"""

import io
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.cloud import vision
import json
import boto3
from IPython.display import display
from PIL import Image, ImageDraw
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures, Direction, DirectionalFinderType
from tqdm import tqdm

# Add this function to preprocess.py
def rotate_image(image, angle):
    """Rotate an image by a given angle.
    
    Args:
        image: numpy array of the image
        angle: angle in degrees to rotate (positive is counterclockwise)
    
    Returns:
        Rotated image as numpy array
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate the center of the image
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to prevent cropping
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform rotation and return
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                  flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    return rotated_image

# 2. Add this resize_image function to your preprocess.py file
def resize_image(image, width=None, height=None, maintain_aspect_ratio=True):
    """
    Resize an image to specific dimensions.
    
    Args:
        image: numpy array of the image
        width: desired width in pixels (if None, will be calculated from height)
        height: desired height in pixels (if None, will be calculated from width)
        maintain_aspect_ratio: if True, preserves aspect ratio
    
    Returns:
        Resized image as numpy array
    """
    # Get current dimensions
    current_height, current_width = image.shape[:2]
    
    # Handle the case where neither width nor height is provided
    if width is None and height is None:
        return image
    
    # Calculate dimensions based on maintain_aspect_ratio
    if maintain_aspect_ratio:
        # If only width is specified, calculate height to maintain aspect ratio
        if width is not None and height is None:
            height = int(current_height * (width / current_width))
        
        # If only height is specified, calculate width to maintain aspect ratio
        elif height is not None and width is None:
            width = int(current_width * (height / current_height))
        
        # If both are specified but we need to maintain aspect ratio,
        # use the smaller scale factor to ensure the image fits within the specified bounds
        elif width is not None and height is not None:
            width_ratio = width / current_width
            height_ratio = height / current_height
            
            # Use the smaller ratio to ensure the image fits within the specified bounds
            if width_ratio < height_ratio:
                height = int(current_height * width_ratio)
            else:
                width = int(current_width * height_ratio)
    else:
        # If not maintaining aspect ratio, ensure both width and height are specified
        if width is None:
            width = current_width
        if height is None:
            height = current_height
    
    # Resize the image
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return resized_image

def process_content(filename,
                    input_folder,
                    output_folder,
                    show_image, 
                    use_google_vision, 
                    use_textract, 
                    verbose):
    input_file = os.path.join(input_folder, filename)
    try:
        if use_google_vision:
            if show_image:
                print("Google Vision Output:")
            else:
                pass
            image = cv2.imread(input_file)
            # Replace the file extension with ".json"
            gcloud_json = os.path.join(output_folder, os.path.splitext(filename)[0] + "_GCloud.json")
            gcloud_boxes(image, gcloud_json, show_image, save_text_to_txt=True)
        else:
            pass
    except:
        print("Error with Cloud Vision")

    try:
        if use_textract:
            print("Running through Textract since use_textract=True")
            if show_image:
                print("Textract Output:")
            else:
                pass
            amazon_json = os.path.join(output_folder, os.path.splitext(filename)[0] + "_Textract.json")
            textract_process_image(input_file, amazon_json, show_image)
        else:
           pass
    except:
        print("Error with Textract")
        
    if verbose:
        print("Setting all parameters=True gives a basic visualization of the outputs of both Cloud Vision, defaulted as the first image, and Textract, the second image. The .txt and .json outputs for both Cloud Vision and Textract are saved in the output_folder. By setting a parameter=False, you can skip that function. For example, if use_textract=False and use_google_vision=True, this will not send the image through Textract, but will send the image through Google Vision.")
    
        

def batch_ocr(input_folder, output_folder, use_google_vision, use_textract):
    # Get the list of image files with valid extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

    # Create a tqdm progress bar
    progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

    for filename in image_files:
        file_path = os.path.join(input_folder, filename)

        process_content(filename, 
                        input_folder, 
                        output_folder, 
                        show_image=False, 
                        use_google_vision=use_google_vision, 
                        use_textract=use_textract, 
                        verbose=False)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    print(f"All images OCR'd. text and JSON files are in folder {output_folder}")


def textract_process_image(image_path, json_output, show_image):
    # Create a Textract Client
    textract = boto3.client('textract')

    # Read the image as bytes
    with open(image_path, 'rb') as document:
        image_bytes = document.read()

    # Call Textract DetectDocumentText to analyze the document
    response = textract.detect_document_text(Document={'Bytes': image_bytes})

    # Save the Textract JSON output to a specified location
    with open(json_output, 'w') as json_output_file:
        json.dump(response, json_output_file, indent=4)

    # Save the detected text to a .txt file
    with open(json_output.replace('.json', '.txt'), 'w') as txt_file:
        for item in response['Blocks']:
            if item['BlockType'] == 'WORD':
                txt_file.write(item['Text'] + '\n')

    # Get the detected text blocks
    blocks = response['Blocks']

    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Check if the image is grayscale and convert to color if needed
    if image.mode == 'L':
        image = image.convert('RGB')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Iterate through the blocks to draw bounding boxes in blue
    for block in blocks:
        if block['BlockType'] == 'WORD':
            polygon = block['Geometry']['Polygon']
            points = [(p['X'] * image.width, p['Y'] * image.height) for p in polygon]
            draw.polygon(points, outline='blue')

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()


#Set default dictionary
default = {
    "left_margin_percent": 15,
    "top_margin_percent": 15, 
    "right_margin_percent": 15,
    "vsplit_percent": 0,
    "hsplit_percent": 0,
    "brightness_factor": 1,
    "contrast_factor": 1,
    "rotation_angle": 0,
    "bottom_margin_percent": 0,
    "resize_width": None,
    "resize_height": None,
    "maintain_aspect_ratio": True, 
    "line_thickness": 3 
}




######
# Extract Tables
###### 

def check_resize_for_textract(image_path, output_folder=None, max_pixels=5000000, max_file_size_bytes=5*1024*1024, quality=80):
    """
    Checks if an image is within Textract's parameters and resizes it if needed.
    
    Args:
        image_path: Path to the input image file
        output_folder: Folder to save temporary resized file (if None, uses original folder)
        max_pixels: Maximum number of pixels allowed (default 5MP for Textract)
        max_file_size_bytes: Maximum file size in bytes (default 5MB for Textract)
        quality: JPEG quality for saved image (1-100)
    
    Returns:
        Path to a properly sized image file (either original or resized temporary file)
    """
    import os
    from PIL import Image
    import io
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Get dimensions and calculate total pixels
        width, height = img.size
        total_pixels = width * height
        
        # Check file size
        file_size = os.path.getsize(image_path)
        
        # Determine if resizing is needed
        needs_resize = False
        needs_convert = img.mode != 'RGB'
        
        # Check if pixel count exceeds maximum
        if total_pixels > max_pixels:
            needs_resize = True
            scale_factor = (max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"Image has {total_pixels} pixels (exceeds {max_pixels})")
            print(f"Resizing from {width}x{height} to {new_width}x{new_height}")
        else:
            new_width, new_height = width, height
        
        # If image is within pixel limit but file size is too large
        if not needs_resize and file_size > max_file_size_bytes:
            needs_resize = True
            # Start with modest reduction in quality
            print(f"Image file size {file_size/1024/1024:.2f}MB exceeds {max_file_size_bytes/1024/1024:.2f}MB")
            
        # If no resize or conversion needed, return original path
        if not needs_resize and not needs_convert:
            print(f"Image '{os.path.basename(image_path)}' is within Textract parameters")
            return image_path
            
        # Setup output path for temporary file
        if output_folder is None:
            output_folder = os.path.dirname(image_path)
            
        # Create temp file name
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        temp_file = os.path.join(output_folder, f"textract_sized_{base_name}.jpg")
        
        # Perform resize if needed
        if needs_resize:
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to RGB if needed
        if needs_convert:
            img = img.convert('RGB')
        
        # Save the image with specified quality
        img.save(temp_file, format='JPEG', quality=quality)
        
        # Check if file size is still too large, reduce quality if needed
        file_size = os.path.getsize(temp_file)
        if file_size > max_file_size_bytes:
            # Try with progressively lower quality until size is acceptable
            for reduced_quality in [70, 60, 50, 40]:
                print(f"File still too large ({file_size/1024/1024:.2f}MB). Reducing quality to {reduced_quality}...")
                img.save(temp_file, format='JPEG', quality=reduced_quality)
                file_size = os.path.getsize(temp_file)
                if file_size <= max_file_size_bytes:
                    break
            
            # If still too large after lowest quality, resize further
            if file_size > max_file_size_bytes:
                print("Still too large after quality reduction, reducing dimensions further...")
                # Calculate new size based on file size ratio
                scale_factor = (max_file_size_bytes / file_size) ** 0.5 * 0.9  # 10% extra reduction for safety
                new_width = int(new_width * scale_factor)
                new_height = int(new_height * scale_factor)
                print(f"Further resizing to {new_width}x{new_height}")
                img = img.resize((new_width, new_height), Image.LANCZOS)
                img.save(temp_file, format='JPEG', quality=60)
        
        print(f"Image prepared for Textract at '{temp_file}'")
        return temp_file
        
    except Exception as e:
        print(f"Error preparing image for Textract: {str(e)}")
        # Return original path if there's an error
        return image_path

def extract_table(extractor, filename, input_folder, output_folder):
    input_file = os.path.join(input_folder, filename)
    
    try:
        # Check and resize the image for Textract if needed
        sized_image_path = check_resize_for_textract(input_file, output_folder)
        
        print(f"Processing image: {filename}")
        
        # Analyze the document using the properly sized image
        document = extractor.analyze_document(
            file_source=sized_image_path,
            features=[TextractFeatures.TABLES],
            save_image=True
        )
        
        # Clean up temporary file if one was created
        if sized_image_path != input_file:
            try:
                os.remove(sized_image_path)
            except:
                pass

        # Show the summary statistics of the detected objects
        print(document)

        # Create variables for the detected words and tables
        words_entity_list = document.words
        table_list = document.tables

        # Load the document image with cv2. This is necessary to draw the bounding boxes
        sing_image = cv2.imread(input_file)
        
        # Check if the image was loaded correctly
        if sing_image is None:
            print(f"Warning: Could not load image for visualization: {input_file}")
            show_boxes = False
        else:
            show_boxes = True
            # Draw the bounding boxes in Textract if the image loaded correctly
            if show_boxes:
                textract_boxes(sing_image, words_entity_list, table_list, show_image=True)
        
        batch_run = input("Do you want to batch extract the Excel tables from images in the input_folder? (y/n): ")
        if batch_run.lower() == "y":
            print("The .xlsx output files will be saved in your output_folder")
            valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
            image_files = [f for f in os.listdir(input_folder) if any(f.lower().endswith(ext) for ext in valid_extensions)]
        
            # Create a tqdm progress bar
            progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')
        
            for img_filename in image_files:
                try:
                    img_file_path = os.path.join(input_folder, img_filename)
                    
                    # Check and resize the image for Textract if needed
                    sized_image_path = check_resize_for_textract(img_file_path, output_folder)
                    
                    # Process with Textract
                    img_document = extractor.analyze_document(
                        file_source=sized_image_path,
                        features=[TextractFeatures.TABLES],
                        save_image=False
                    )
                    
                    # Clean up temporary file if one was created
                    if sized_image_path != img_file_path:
                        try:
                            os.remove(sized_image_path)
                        except:
                            pass
                    
                    # Save Excel if tables found
                    file_name, current_extension = os.path.splitext(img_filename)
                    new_filename = file_name + '.xlsx' 
                    output_path = os.path.join(output_folder, new_filename)
                    
                    if img_document.tables and len(img_document.tables) > 0:
                        img_document.tables[0].to_excel(output_path)
                        print(f"Table extracted from {img_filename}")
                    else:
                        print(f"No tables found in {img_filename}")
                
                except Exception as e:
                    print(f"Error processing {img_filename}: {str(e)}")
                
                # Update the progress bar
                progress_bar.update(1)

            # Close the progress bar
            progress_bar.close()
        
        else:
            # Ask if user wants to extract Excel table from the current image
            single_extract = input("Do you want to extract the Excel table from this image? (y/n): ")
            if single_extract.lower() == "y":
                try:
                    file_name, current_extension = os.path.splitext(filename)
                    new_filename = file_name + '.xlsx'
                    output_path = os.path.join(output_folder, new_filename)
                    
                    if document.tables and len(document.tables) > 0:
                        document.tables[0].to_excel(output_path)
                        print(f"Excel table saved as {new_filename} in {output_folder}")
                    else:
                        print("No tables found in the image")
                except Exception as e:
                    print(f"Error extracting table: {str(e)}")
    
    except Exception as e:
        print(f"Error processing image {filename}: {str(e)}")
        print("\nTry using a smaller image or one with clearer table structure.")

######
#Preprocess only
######

def preprocess_image(filename, input_folder, output_folder, 
                    left_margin_percent, top_margin_percent, right_margin_percent, 
                    vsplit_percent, hsplit_percent, brightness_factor, contrast_factor, 
                    rotation_angle, bottom_margin_percent=0, resize_width=None, resize_height=None,
                    maintain_aspect_ratio=True, line_thickness=3):
   
    print(f"PREPROCESSING: vsplit_percent={vsplit_percent}")
    print(f"PREPROCESSING: default values={default}") 
    print(f"preprocess_image received rotation_angle: {rotation_angle}")
    print(f"Type of rotation_angle: {type(rotation_angle)}")

    # Get the height and width of the input image
    image = cv2.imread(os.path.join(input_folder, filename))
    
    # Apply rotation first if needed
    if rotation_angle != 0:
        image = rotate_image(image, rotation_angle)
    
    # Get dimensions after rotation
    height, width = image.shape[0:2]
    
    split_question = input('Do you want to split this image into two separate images? (y/n):')
    if split_question.lower() =='n':
        vsplit_percent = 0
        hsplit_percent = 0
        pass
    if split_question.lower() == 'y':
        vert_horiz = input('Do you want to split it Vertically or Horizontally? (v/h)')
        if vert_horiz.lower() == 'v' and vsplit_percent ==0:
            default['vsplit_percent'] = 50
            vsplit_percent = 50
            hsplit_percent = 0
        if vert_horiz.lower() == 'v' and vsplit_percent !=0:
            hsplit_percent = 0            
        if vert_horiz.lower() == 'h' and hsplit_percent ==0:            
            vsplit_percent = 0
            hsplit_percent = 50
            default['hsplit_percent'] = 50
        if vert_horiz.lower() == 'h' and hsplit_percent !=0:
            vsplit_percent = 0    
        else:
            pass

    # Calculate margin values based on percentages
    LeftMargin = int(width * (left_margin_percent / 100))
    TopMargin = int(height * (top_margin_percent / 100))
    RightMargin = int(width * (1 - right_margin_percent / 100))
    BottomMargin = int(height * (1 - bottom_margin_percent / 100)) if bottom_margin_percent > 0 else height
    
    # Calculate split lines based on percentages
    VSplit = int(width * (vsplit_percent / 100))
    HSplit = int(height * (hsplit_percent / 100))

    # Create a copy of the image to draw margin lines
    MarginTest = image.copy()
    
    # Process image (but skip rotation since we've already done it)
    MarginTest = grayscale(MarginTest)
    if brightness_factor != 1.0:
        MarginTest = adjust_brightness(MarginTest, brightness_factor)
    if contrast_factor != 1.0:
        MarginTest = adjust_contrast(MarginTest, contrast_factor)

    # Draw margin lines
    cv2.line(MarginTest, (LeftMargin, TopMargin), (RightMargin, TopMargin), (255, 0, 0), line_thickness)
    cv2.line(MarginTest, (LeftMargin, TopMargin), (LeftMargin, BottomMargin), (255, 0, 0), line_thickness)
    cv2.line(MarginTest, (LeftMargin, BottomMargin), (RightMargin, BottomMargin), (255, 0, 0), line_thickness)
    cv2.line(MarginTest, (RightMargin, TopMargin), (RightMargin, BottomMargin), (255, 0, 0), line_thickness)
    
    # Draw split lines
    if vsplit_percent != 0:
        cv2.line(MarginTest, (VSplit, 0), (VSplit, height), (0, 0, 255), line_thickness)
    if hsplit_percent != 0:
        cv2.line(MarginTest, (0, HSplit), (width, HSplit), (0, 0, 255), line_thickness)
        

    # Display the image with the margins
    plt.figure(figsize=(10, 20))
    plt.imshow(MarginTest, cmap='gray')
    plt.show()
    
    # Ask the user if they are satisfied with the result
    satisfaction = input("Are you satisfied with the outline you see? (y/n): ")
    if satisfaction.lower() == 'y':
        # Create a subfolder 'modified_images' if it doesn't exist
        modified_images_folder = os.path.join(output_folder, 'modified_images')
        os.makedirs(modified_images_folder, exist_ok=True)
        print("Here is an output based on these parameters")
        if vsplit_percent == 0 and hsplit_percent == 0:
            # Apply blank margins
            image = blank_margins(image, left_margin_percent, top_margin_percent, right_margin_percent)
            
            # Apply bottom margin if specified
            if bottom_margin_percent > 0:
                # White color
                White = (255, 255, 255)
                # Bottom margin
                cv2.rectangle(image, (0, BottomMargin), (width, height), White, -1)
            
            # Process image
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
            
            # Apply resize if specified
            if resize_width is not None or resize_height is not None:
                image = resize_image(image, resize_width, resize_height)
            
            output_path = os.path.join(modified_images_folder, 'modified_' + filename)
            cv2.imwrite(output_path, image)
            plt.figure(figsize=(10, 20))
            plt.imshow(image, cmap='gray')
            plt.show()
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")
                       
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    
                    # Apply rotation if needed
                    if rotation_angle != 0:
                        image = rotate_image(image, rotation_angle)
                    
                    # Apply blank margins
                    image = blank_margins(image, left_margin_percent, top_margin_percent, right_margin_percent)
                    
                    # Apply bottom margin if specified
                    if bottom_margin_percent > 0:
                        height, width = image.shape[:2]
                        BottomMargin = int(height * (1 - bottom_margin_percent / 100))
                        # White color
                        White = (255, 255, 255)
                        # Bottom margin
                        cv2.rectangle(image, (0, BottomMargin), (width, height), White, -1)
                    
                    # Process image
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
                    
                    # Apply resize if specified
                    if resize_width is not None or resize_height is not None:
                        image = resize_image(image, resize_width, resize_height)
                    
                    output_path = os.path.join(modified_images_folder, 'modified_' + filename)
                    cv2.imwrite(output_path, image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            return image
            
        # The rest of the function for split cases remains unchanged
        # ...
        if vsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
            left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=True)
            
            # Apply resize if specified
            if resize_width is not None or resize_height is not None:
                left_image = resize_image(left_image, resize_width, resize_height)
                right_image = resize_image(right_image, resize_width, resize_height)
            
            L_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
            cv2.imwrite(L_output_path, left_image)
            R_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
            cv2.imwrite(R_output_path, right_image)
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    
                    # Apply rotation if needed
                    if rotation_angle != 0:
                        image = rotate_image(image, rotation_angle)
                    
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
                    left_image, right_image = two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=False)
                    
                    # Apply resize if specified
                    if resize_width is not None or resize_height is not None:
                        left_image = resize_image(left_image, resize_width, resize_height)
                        right_image = resize_image(right_image, resize_width, resize_height)
                    
                    L_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
                    cv2.imwrite(L_output_path, left_image)
                    R_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
                    cv2.imwrite(R_output_path, right_image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            return left_image, right_image
            
        if hsplit_percent !=0:
            image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
            top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=True)
            
            # Apply resize if specified
            if resize_width is not None or resize_height is not None:
                top_image = resize_image(top_image, resize_width, resize_height)
                bottom_image = resize_image(bottom_image, resize_width, resize_height)
            
            T_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
            cv2.imwrite(T_output_path, top_image)
            B_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
            cv2.imwrite(B_output_path, bottom_image)
            print("Preprocessed images are saved in a subfolder of your output folder called 'modified_images'.")            
            
            
            batch_run = input("Do you want to batch run this preprocessing routine on the entire input folder? (y/n): ")
            if batch_run.lower() =="y":
                valid_extensions = ['.jpg', '.jpeg', '.png', '.gif']
                image_files = [filename for filename in os.listdir(input_folder) if any(filename.lower().endswith(ext) for ext in valid_extensions)]

                # Create a tqdm progress bar
                progress_bar = tqdm(total=len(image_files), desc='Processing Images', unit='image')

                for filename in image_files:
                    file_path = os.path.join(input_folder, filename)
                    
                    image = cv2.imread(file_path)
                    
                    # Apply rotation if needed
                    if rotation_angle != 0:
                        image = rotate_image(image, rotation_angle)
                    
                    image = process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle)
                    top_image, bottom_image = two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=False)
                    
                    # Apply resize if specified
                    if resize_width is not None or resize_height is not None:
                        top_image = resize_image(top_image, resize_width, resize_height)
                        bottom_image = resize_image(bottom_image, resize_width, resize_height)
                    
                    T_output_path = os.path.join(modified_images_folder, 'modified_1_' + filename)
                    cv2.imwrite(T_output_path, top_image)
                    B_output_path = os.path.join(modified_images_folder, 'modified_2_' + filename)
                    cv2.imwrite(B_output_path, bottom_image)
                    
                    # Update the progress bar
                    progress_bar.update(1)

                # Close the progress bar
                progress_bar.close()
            
            
            
            return top_image, bottom_image
    else:
        # If not satisfied, stop the script
            print("Current settings are ", default)
            print("You can modify these settings in the modification cell above.")
            print("Suppose you want to change the side margins to be 8% of the page. Then type: pp.default['left_margin_percent'] = 8")









#Define a program that splits an image vertically into two separate images and adds white space where the split occurs




def two_split_vert(image, vsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=True):
    # Get the height and width of the input image
    original_height, original_width = image.shape[0:2]
    print(f"ORIGINAL IMAGE SIZE: height={original_height}, width={original_width}")
    
    # Calculate margin values based on percentages
    LeftMargin = int(original_width * (left_margin_percent / 100))
    TopMargin = int(original_height * (top_margin_percent / 100))
    RightMargin = int(original_width * (1 - right_margin_percent / 100)) 
    BottomMargin = int(original_height - TopMargin)
    
    print(f"CALCULATED MARGINS: Left={LeftMargin}, Top={TopMargin}, Right={RightMargin}, Bottom={BottomMargin}")

    # Insert the desired color for the rectangle (white in this case)
    White = (255, 255, 255)

    # White out the portions outside of the margins, but DON'T crop
    # Just create a copy of the image to avoid modifying the original
    processed_image = image.copy()
    
    # Whiteout only if margins are set (rather than actually cropping)
    if left_margin_percent > 0:
        # Left margin
        cv2.rectangle(processed_image, (0, 0), (LeftMargin, original_height), White, -1)
    
    if right_margin_percent > 0:
        # Right margin
        cv2.rectangle(processed_image, (RightMargin, 0), (original_width, original_height), White, -1)
        
    if top_margin_percent > 0:
        # Top margin
        cv2.rectangle(processed_image, (0, 0), (original_width, TopMargin), White, -1)
    
    # We're NOT whiting out the bottom - we want to keep all content
        
    # Skip the cropping step entirely
    # Calculate middle for splitting (based on visible area between margins)
    middle = int(original_width * (vsplit_percent / 100))
    print(f"SPLIT POINT: middle={middle}")
    
    # Split the ORIGINAL image into two separate images
    left_image = processed_image[:, :middle]
    right_image = processed_image[:, middle:]
    
    left_height, left_width = left_image.shape[0:2]
    right_height, right_width = right_image.shape[0:2]
    print(f"SPLIT IMAGES: left={left_height}x{left_width}, right={right_height}x{right_width}")

    # Add padding to the split
    padding = np.zeros((original_height, split_padding, 3), dtype=np.uint8)
    padding = 255 - padding  # Make it white
    
    left_image = np.hstack((left_image, padding))
    right_image = np.hstack((padding, right_image))
    
    final_left_height, final_left_width = left_image.shape[0:2]
    final_right_height, final_right_width = right_image.shape[0:2]
    print(f"FINAL IMAGES: left={final_left_height}x{final_left_width}, right={final_right_height}x{final_right_width}")

    if show_image:
        # Display the images
        plt.figure(figsize=(10, 20))
        plt.subplot(121), plt.imshow(left_image), plt.title('Left Image')
        plt.subplot(122), plt.imshow(right_image), plt.title('Right Image')
        plt.show()

    return left_image, right_image
# Example usage:
# Load your image and call the function with it
# image = cv2.imread('your_image.png')
# left_image, right_image = white_out_and_crop(image, left_margin_percent=30, top_margin_percent=5, split_padding=100, show_image=True)





#Define a program that splits an image horizontally into two separate images and adds white space where the split occurs

def two_split_horiz(image, hsplit_percent, left_margin_percent, top_margin_percent, right_margin_percent, split_padding=100, show_image=True):
    # Get the height and width of the input image
    height, width = image.shape[0:2]

    # Calculate margin values based on percentages
    TopMargin = int(height * (top_margin_percent / 100))
    BottomMargin = int(height - TopMargin)
    LeftMargin = int(width * (left_margin_percent / 100))
    RightMargin = int(width * (1 - right_margin_percent / 100)) 

    # Insert the desired color for the rectangle (white in this case)
    White = (255, 255, 255)

    # Define rectangles for the portions outside of the margins
    BLMargin = (0, BottomMargin)
    BRCorner = (width, height)
    TLMargin = (LeftMargin, 0)
    TRCorner = (width, TopMargin)
    BRMargin = (RightMargin, height)

    # White out the portions outside of the margins
    image = cv2.rectangle(image, BLMargin, BRCorner, White, -1)
    image = cv2.rectangle(image, TLMargin, BLMargin, White, -1)
    image = cv2.rectangle(image, TLMargin, TRCorner, White, -1)
    image = cv2.rectangle(image, BRMargin, TRCorner, White, -1)

    # Crop the image to give the illusion of normal page margins
    image = image[int(1 - BottomMargin * 1.1):int(BottomMargin * 1.1), int(1 - RightMargin * 1.1):int(RightMargin * 1.1)]

    height, width = image.shape[0:2]
    
    # Split the image into two separate images with padding
    middle = int(height * (hsplit_percent / 100))
    top_image = image[:middle, :]
    bottom_image = image[middle:, :]

    # Add padding to the split
    padding = np.zeros((split_padding, width, 3), dtype=np.uint8)
    padding = 255 - padding
    top_image = np.vstack((top_image, padding))
    bottom_image = np.vstack((padding, bottom_image))

    if show_image:
        # Display the images
        plt.figure(figsize=(10, 20))
        plt.subplot(121), plt.imshow(top_image), plt.title('Top Image')
        plt.subplot(122), plt.imshow(bottom_image), plt.title('Bottom Image')
        plt.show()

    return top_image, bottom_image

# Example usage:
# Load your image and call the function with it
# image = cv2.imread('your_image.png')
# top_image, bottom_image = white_out_and_crop(image, top_margin_percent=5, left_margin_percent=30, split_padding=100, show_image=True)

        
def textract_boxes(image, words_entity_list, table_list, show_image=True):
    # Get image dimensions
    height, width, _ = image.shape
    

    # Iterate over word entities and draw rectangles
    for word_entity in words_entity_list:
        # Accessing the 'BoundingBox' attribute directly
        bounding_box = word_entity.bbox
        
        xmin = int(bounding_box.x * width)
        ymin = int(bounding_box.y * height)
        xmax = int((bounding_box.x + bounding_box.width) * width)
        ymax = int((bounding_box.y + bounding_box.height) * height)

        # Draw a rectangle on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()
        
        satisfaction = input("Do you want to see the table output? (y/n): ")
        if satisfaction.lower() == 'y':
                for table in table_list:
                    df=table.to_pandas()
                    display(df)
        else:
            print('no')








def gcloud_boxes(image, output_file, show_image, save_text_to_txt):
    # Initialize the Google Cloud Vision client
    client = vision.ImageAnnotatorClient()

    # Convert the input image to encoded PNG format
    _, encoded_image = cv2.imencode('.png', image)

    # Create a Vision API image object
    api_image = vision.Image(content=encoded_image.tobytes())

    # Perform text detection
    response = client.text_detection(image=api_image)
    texts = response.text_annotations

    for text in texts:
        vertices = np.array([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])

        # Calculate bounding box coordinates
        xmin, xmax = min(vertices[:, 0]), max(vertices[:, 0])
        ymin, ymax = min(vertices[:, 1]), max(vertices[:, 1])

        # Draw bounding boxes on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

    if response.error.message:
        print(response.error.message)

    if show_image:
        # Display the image with bounding boxes
        plt.figure(figsize=(10, 20))
        plt.imshow(image)
        plt.show()

    bounding_boxes = []

    # Extract bounding box and text information
    for text in texts:
        vertices = text.bounding_poly.vertices
        box_data = {
            "text": text.description,
            "bounding_box": {
                "vertices": [
                    {"x": vertex.x, "y": vertex.y}
                    for vertex in vertices
                ]
            }
        }
        bounding_boxes.append(box_data)

    # Save the bounding box data to a JSON file
    with open(output_file, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=2)

    if save_text_to_txt:
        # Save the extracted text to a .txt file
        with open(output_file.replace('.json', '.txt'), 'w') as txt_file:
            for text in texts:
                txt_file.write(text.description + '\n')








# Define a function to adjust brightness
def adjust_brightness(image, brightness_factor):
    # Adjust brightness using cv2.convertScaleAbs
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return adjusted_image

# Define a function to adjust contrast
def adjust_contrast(image, contrast_factor):
    # Adjust contrast using cv2.convertScaleAbs
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    return adjusted_image

# Define a function to convert the image to grayscale
def grayscale(image):
    # Convert the image to grayscale
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    # Compute the per-channel means
    b_mean = b.mean()
    g_mean = g.mean()
    r_mean = r.mean()

    # Compute the scaling factors
    k = (b_mean + g_mean + r_mean) / 3

    # Apply the scaling factors to each channel
    corrected_b = (b * k / b_mean).clip(0, 255).astype('uint8')
    corrected_g = (g * k / g_mean).clip(0, 255).astype('uint8')
    corrected_r = (r * k / r_mean).clip(0, 255).astype('uint8')

    # Merge the corrected channels
    corrected_image = cv2.merge((corrected_b, corrected_g, corrected_r))
    
    # Convert the corrected image to grayscale
    gray_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
    return gray_image

#Whiting out margins and cropping 


def blank_margins(image, left_margin_percent, top_margin_percent, right_margin_percent):
    """
    Improved version of blank_margins that properly handles zero margins.
    
    Args:
        image: numpy array of the image
        left_margin_percent: percentage of image width to blank out from left (0-100)
        top_margin_percent: percentage of image height to blank out from top (0-100)
        right_margin_percent: percentage of image width to blank out from right (0-100)
    
    Returns:
        Image with margins filled with white
    """
    # Get the height and width of the input image
    height, width = image.shape[:2]

    # Create a copy of the image to avoid modifying the original
    result = image.copy()
    
    # Insert the desired color for the rectangle (white)
    White = (255, 255, 255)

    # Calculate margin positions (only if the margin percent > 0)
    LeftMargin = int(width * (left_margin_percent / 100)) if left_margin_percent > 0 else 0
    TopMargin = int(height * (top_margin_percent / 100)) if top_margin_percent > 0 else 0
    RightMargin = int(width * (right_margin_percent / 100)) if right_margin_percent > 0 else 0
    
    # Only draw white rectangles for margins that are > 0%
    if left_margin_percent > 0:
        # Left margin
        cv2.rectangle(result, (0, 0), (LeftMargin, height), White, -1)
    
    if top_margin_percent > 0:
        # Top margin
        cv2.rectangle(result, (0, 0), (width, TopMargin), White, -1)
    
    if right_margin_percent > 0:
        # Right margin
        cv2.rectangle(result, (width - RightMargin, 0), (width, height), White, -1)
    
    # Note: This version doesn't crop the image afterward like the original did
    # This avoids the issue of cropping when margins are set to 0
    return result


# Define a function to process an image with options to show the output
def process_image(image, brightness_factor, contrast_factor, left_margin_percent, top_margin_percent, right_margin_percent, vsplit_percent, hsplit_percent, rotation_angle, resize_width=None, resize_height=None): 
    
    # Convert to grayscale
    image = grayscale(image)   

    # Apply brightness adjustment
    if brightness_factor != 1.0:
        image = adjust_brightness(image, brightness_factor)

    # Apply contrast adjustment
    if contrast_factor != 1.0:
        image = adjust_contrast(image, contrast_factor)
    
    # Add resize functionality
    if resize_width is not None or resize_height is not None:
        image = resize_image(image, resize_width, resize_height)
    
    return image

