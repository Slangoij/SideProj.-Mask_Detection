from starlette.responses import HTMLResponse
from starlette.applications import Starlette
from jinja2 import Template

from server_processing import img_processer
import uvicorn

template = """\
<!DOCTYPE HTML>
<html>
<head>
    <title>Streamer</title>
</head>
<body>
    <video autoplay hidden=true></video>
    <img src="">
    <script type = "text/javascript">
    
        const video = document.querySelector('video');
        const img = document.querySelector('img');
        navigator.mediaDevices.getUserMedia({video: {width: 426, height: 240}}).then((stream) => video.srcObject = stream);

        const getFrame = () => {
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            const data = canvas.toDataURL('image/png');
            return data;
        }

        const WS_URL = 'ws://localhost:5500/ws';   // <<<<<<< 이 부분 port번호 맞추기
        const FPS = 3;
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log(`Connected to ${WS_URL}`);
            setInterval(() => {
                ws.send(getFrame());
            }, 1000 / FPS);
        }

        ws.onmessage = message => {
            // set the base64 string to the src tag of the image
            img.src = message.data;
            console.log(img.src)
        }
    </script>
</body>
</html>
"""

app = Starlette()

@app.route('/')
async def homepage(request):
    return HTMLResponse(Template(template).render())


@app.websocket_route('/ws')
async def websocket_endpoint(websocket):
    await websocket.accept()
    # Process incoming messages
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(img_processer(data))
    await websocket.close()


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5500)  # <<< 32번 줄 포트 번호랑 맞추기