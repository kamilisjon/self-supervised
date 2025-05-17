import os, csv

# Instructions:
# 1. Download data from https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task

class PneumothoraxData:
  def __init__(self, data_path):
    self.data_dict = {"0": [], "1": []}
    with open(os.path.join(data_path, "train_data.csv"), 'r') as f:
      for row in csv.DictReader(f): self.data_dict[row["target"]].append(os.path.join(data_path, row["file_name"]))

if __name__ == "__main__":
  pneumo_data = PneumothoraxData(os.path.expanduser("~/Downloads/archive/small_train_data_set/small_train_data_set"))
  print(pneumo_data.data_dict)