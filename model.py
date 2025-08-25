from ultralytics import RTDETR


model = RTDETR()
def get_video_predictions(video_path):
    answer = model.predict(video_path, stream=True, batch=8, vid_stride=5)
    for r in answer:
        yield r


