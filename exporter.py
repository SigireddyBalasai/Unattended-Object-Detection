from ultralytics import RTDETR
import os

model = RTDETR("rtdetr-l.pt")
model.export(format="engine")  # export to TensorRT engine with dynamic shape
os.makedirs("model_repository/rtdetr_tensorrt/1", exist_ok=True)
os.rename("rtdetr-l.engine", "model_repository/rtdetr_tensorrt/1/model.plan")
