import os, csv, random

from tinygrad.helpers import tqdm
import cv2
import albumentations as A

# Instructions:
# 1. Download data from https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task

random.seed(0)

class PneumothoraxData:
  def __init__(self, data_path, split_ratio=0.8):
    self.data_dict = {"0": [], "1": []}
    with open(os.path.join(data_path, "train_data.csv"), 'r') as f:
      for row in csv.DictReader(f): self.data_dict[row["target"]].append(os.path.join(data_path, row["file_name"]))

    self.split_dict = {"train": [], "val": []}
    for target in self.data_dict.keys():
      paths = self.data_dict[target]
      random.shuffle(paths)
      split_idx = int(len(paths) * split_ratio)
      self.split_dict["train"], self.split_dict["val"] = paths[:split_idx], paths[split_idx:]

  def generate_blurred_images(self, output_base_path):
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    class_names = [f"blur_{int(sigma*10) if sigma.is_integer() else str(sigma).replace('.', '')}" for sigma in sigma_values]
    for split in ["train", "val"]:
      for img_path in tqdm(self.split_dict[split], desc=f"Generating blurred data. Set: {split}"):
        img = cv2.imread(img_path)
        for sigma, class_name in zip(sigma_values, class_names):
          augmented = A.GaussianBlur(blur_limit=0, sigma_limit=(sigma, sigma), p=1)(image=img)["image"]
          output_path = os.path.join(output_base_path, split, class_name, os.path.basename(img_path))
          os.makedirs(os.path.dirname(output_path), exist_ok=True)
          cv2.imwrite(str(output_path), augmented)

if __name__ == "__main__":
  pneumo_data = PneumothoraxData(os.path.expanduser("~/Downloads/archive/small_train_data_set/small_train_data_set"))
  print("Data stats")
  for k, v in list(pneumo_data.data_dict.items()): print(f"Pneumothorax class {k}. Images count: {len(v)}")
  for k, v in list(pneumo_data.split_dict.items()): print(f"Split {k}. Images count: {len(v)}")
  pneumo_data.generate_blurred_images(os.path.expanduser("~/Downloads/archive/blurred_data"))