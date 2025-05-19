import os
from PIL import Image

def resize_images_in_folder(folder_path, target_width=1920, target_height=1080):
    """
    Resizes all images in a specified folder to the target dimensions based on aspect ratio.

    Args:
        folder_path (str): The path to the folder containing images.
        target_width (int): The target width in pixels for landscape images.
        target_height (int): The target height in pixels for landscape images.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                original_width, original_height = img.size

                # Determine target dimensions based on aspect ratio
                if original_width > original_height:
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                else:
                    resized_img = img.resize((target_height, target_width), Image.Resampling.LANCZOS)

                # Save the image, overwriting the original
                resized_img.save(file_path)
                print(f"Resized '{filename}' to {resized_img.size[0]}x{resized_img.size[1]}")
            except FileNotFoundError:
                print(f"Error: File '{file_path}' not found during processing (should not happen if os.listdir worked).")
            except IOError:
                print(f"Error: Could not open or read image file '{filename}'. It might be corrupted or not a valid image.")
            except Exception as e:
                print(f"An unexpected error occurred with file '{filename}': {e}")

if __name__ == "__main__":
    # 重要: 请将 'your_image_folder_path' 替换为你的图片文件夹的实际路径
    image_folder = 'pics' 
    if not os.path.exists(image_folder):
        print(f"Error: The specified folder '{image_folder}' does not exist.")
        exit(1)
    resize_images_in_folder(image_folder)
    print("Image resizing process complete.")
    
        