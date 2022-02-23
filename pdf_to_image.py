from pdf2image import convert_from_path
import os
import shutil
from get_values_from_config_file import dataset_folder_path, final_images_folder, image_extension, raw_files_folder
from get_values_from_config_file import poppler_path


# Get list of all pdf files and image files present in the sub-folders
def get_list_of_files_from_sub_folders(input_path):
    file_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


# Convert first page of pdf to image
def convert_pdf_first_page_to_image(pdf_path, output_path):
    pdf_pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    pdf_pages[0].save(output_path + '.jpg', 'JPEG')


# Move converted pdf to images or images to input file
def move_files_from_raw_to_input_folder():
    # Create Images folder to be used for model training
    image_folder_path = os.path.join(dataset_folder_path, final_images_folder)
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    # File list from all sub-folders
    raw_input_path = os.path.join(dataset_folder_path, raw_files_folder)
    files_list = get_list_of_files_from_sub_folders(raw_input_path)
    for file in files_list:
        class_name = file.split("\\")[-2]  # Get the class name from folders
        file_name = file.split("\\")[-1]  # Get the filename with extension
        file_name_without_extension = os.path.splitext(file_name)[0]  # Split the filename from the extension
        extension = os.path.splitext(file_name)[1]  # Get the extension of the file
        image_save_path = os.path.join(image_folder_path, class_name)  # Final saving path for each image
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        if extension == '.pdf':
            convert_pdf_first_page_to_image(file, f'{image_save_path}/{file_name_without_extension}')
        elif extension in image_extension:
            shutil.copy(file, image_save_path)
        else:
            print(f"{file_name} in folder {class_name} does not have a valid file extension!!")

    print(f"The files have been moved from {raw_files_folder} to {dataset_folder_path}")


if __name__ == '__main__':
    move_files_from_raw_to_input_folder()
