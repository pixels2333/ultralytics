from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("ultralytics\\cfg\\models\\11\\myyolo.yaml")
model = YOLO("ultralytics\\cfg\\models\\11\\myyolo.yaml").load(
    "yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="mnist160",
    # data="/kaggle/input/tomatoleaf/tomato",
    epochs=100,  # 训练轮数
    batch=32,  # 每批次的样本数量
    device="cpu",  # 使用 CPU
    # device="[0,1]",
    cache=False,  # 禁用缓存
    # workers=0  # 禁用多线程数据加载
)
