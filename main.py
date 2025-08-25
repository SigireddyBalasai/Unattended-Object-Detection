from flask import Flask
from flask_websockets import WebSocket, WebSockets
from model import get_video_predictions
import json

app = Flask(__name__)
websockets = WebSockets(app)

@websockets.route("/ws")
def websocket_route(ws: WebSocket) -> None:
    while True:
        data = next(ws.iter_data())
        print("Received:", data)
        data = json.loads(data)
        if "video_path" in data:
            video_path = data["video_path"]
            for prediction in get_video_predictions(video_path):
                ws.send(str(prediction))
            break
        else:
            ws.send("Please send a JSON object with 'video_path'.")

@app.route("/")
def main():
    return "Hello, World!"

app.run(host="0.0.0.0", port=6969)