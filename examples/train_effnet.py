import traceback
import time
from multiprocessing import Process, Queue, Event
from PIL import Image
import albumentations as A
import os
import numpy as np
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv, trange
from tinygrad.tensor import Tensor
from self_supervised.models.efficientnet import EfficientNet


class ImageLoader:
  def __init__(self, data_path: str, preprocess_func: callable, batch_size: int, num_workers: int):
    self.img_paths = [os.path.join(root, f) for root, _, files in os.walk(data_path) for f in files]
    self.batch_size = batch_size
    self.preprocess_func = preprocess_func
    self.__q = Queue(16)
    self.__shutdown_event = Event()
    self.__workers: list[Process] = [Process(target=self.__fetch_batch, args=(self.__q, self.__shutdown_event, self.img_paths)) for _ in range(num_workers)]
    for w in self.__workers: w.daemon = True

  def start(self):
    for w in self.__workers: w.start()

  def stop(self):
    self.__shutdown_event.set()
    for w in self.__workers: w.join()

  def get(self): return self.__q.get()

  def __fetch_batch(self, q, shutdown_event, img_paths):
    while not shutdown_event.is_set():
      try:
        q.put([self.preprocess_func(img_paths[i]) for i in np.random.permutation(len(img_paths))[:self.batch_size]])
      except Exception:
          traceback.print_exc()

def get_preprocess_func(data_path):
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToFloat(max_value=255)
    ])
    def preprocess_func(img_path):
        img = np.array(Image.open(img_path).convert("RGB"))
        img = transform(image=img)["image"]
        class_name = os.path.basename(os.path.dirname(img_path))
        class_idx = class_to_idx[class_name]
        return img, class_idx
    return preprocess_func

if __name__ == "__main__":
  classes = 6
  model = EfficientNet(0, classes=1000, has_se=True)
  model.load_from_pretrained()
  model._fc = Tensor.glorot_uniform(1280, classes)
  model._fc_bias = Tensor.zeros(classes)
  parameters = get_parameters(model)
  print("parameter count", len(parameters))
  optimizer = optim.Adam(parameters, lr=0.001)
  BS, steps = getenv("BS", 16), getenv("STEPS", 2048)
  print(f"training with batch size {BS} for {steps} steps")

  train_loader = ImageLoader("/home/kamilis/datasets/pretex_blur/train", 
                             preprocess_func=get_preprocess_func("/home/kamilis/datasets/pretex_blur/train"), batch_size=BS, num_workers=4)
  train_loader.start()

  with Tensor.train():
    for i in (t := trange(steps)):
      batch = train_loader.get()
      X = np.stack([item[0] for item in batch]).transpose(0, 3, 1, 2)
      Y = np.array([item[1] for item in batch])

      st = time.time()
      out = model.forward(Tensor(X.astype(np.float32), requires_grad=False))
      fp_time = (time.time()-st)*1000.0

      # One-hot encode labels for cross-entropy
      y = np.zeros((BS, classes), dtype=np.float32)
      y[range(BS), Y] = 1.0  # Positive class gets 1
      y = Tensor(y, requires_grad=False)
      loss = out.log_softmax().mul(y).mean()

      optimizer.zero_grad()

      st = time.time()
      loss.backward()
      bp_time = (time.time()-st)*1000.0

      st = time.time()
      optimizer.step()
      opt_time = (time.time()-st)*1000.0

      st = time.time()
      loss = loss.numpy()
      cat = out.argmax(axis=1).numpy()
      accuracy = (cat == Y).mean()
      finish_time = (time.time()-st)*1000.0

      # printing
      t.set_description("loss %.2f accuracy %.2f -- %.2f + %.2f + %.2f + %.2f = %.2f" %
        (loss, accuracy,
        fp_time, bp_time, opt_time, finish_time,
        fp_time + bp_time + opt_time + finish_time))

      del out, y, loss
  train_loader.stop()