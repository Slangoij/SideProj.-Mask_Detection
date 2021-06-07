from starlette.responses import HTMLResponse
from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from server_processing import img_processer
import uvicorn

app = Starlette()
templates = Jinja2Templates(directory='templates')

@app.route('/')
async def homepage(request):
    return templates.TemplateResponse('home.html', {'request': request})

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