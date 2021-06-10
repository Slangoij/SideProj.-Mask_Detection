from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.routing import Mount
import uvicorn

from server.server_processing import img_processer

templates = Jinja2Templates(directory='templates')
routes = [
    Mount('/assets', app=StaticFiles(directory='assets'), name='assets'),
]

app = Starlette(routes=routes)

@app.route('/')
async def homepage(request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.websocket_route('/ws')
async def websocket_endpoint(websocket):
    await websocket.accept()
    # Process incoming messages
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(img_processer(data))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5500)