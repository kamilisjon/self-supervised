import os
import cv2
import matplotlib.pyplot as plt

def visualize_classification_data(folder_path, num_images=5):
    class_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    fig, axes = plt.subplots(len(class_folders), num_images, figsize=(num_images * 3, len(class_folders) * 3))
    
    for i, class_name in enumerate(class_folders):
        class_path = os.path.join(folder_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_files = image_files[:num_images]
        
        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if len(class_folders) == 1:
                axes[j].imshow(img)
                axes[j].set_title(class_name if j == 0 else "")
                axes[j].axis('off')
            else:
                axes[i, j].imshow(img)
                axes[i, j].set_title(class_name if j == 0 else "")
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('classification_visualization.png')

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing class folders: ")
    visualize_classification_data(os.path.expanduser(folder_path))
    print("Visualization saved as 'class_visualization.png' in the input folder")