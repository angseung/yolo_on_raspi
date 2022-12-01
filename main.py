import torch
import yaml
from torchinfo import summary
from models.raspi_yolo import yolo

with open("./models/voc/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

model = yolo(config)
width = 384
height = 384

dummy_input = torch.randn(1, 3, width, height)
print(summary(model, (1, 3, width, height)))
