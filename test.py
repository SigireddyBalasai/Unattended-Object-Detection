import json
import websocket

# Path to a video file in the ABODA folder
VIDEO_PATH = "ABODA/video1.avi"

# WebSocket server URL
WS_URL = "ws://localhost:6969/ws"

def on_message(ws, message):
    print("Received:", message)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed")

def on_open(ws):
    # Send the video path as a JSON object
    ws.send(json.dumps({"video_path": VIDEO_PATH}))

if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
