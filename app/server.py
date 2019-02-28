from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.text import *

model_file_url = 'https://www.dropbox.com/s/fsk3wfwpivwimlw/imdb.pth?dl=1'
model_file_name = 'imdb'
encoder_file_url = 'https://www.dropbox.com/s/08soznj8n4ccdgh/imdb_enc.pth?dl=1'
encoder_file_name = 'imdb_enc'

classes = ['neg', 'pos']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    await download_file(encoder_file_url, path/'models'/f'{encoder_file_name}.pth')
    data_clas = TextList.from_folder(path)
    #data_clas.save('tmp_clas')
    #data_clas = TextClasDataBunch.load(path, 'tmp_clas')
    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
    learn.load_encoder(encoder_file_name)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    text = await (data['file'].read())
    prediction = learn.predict(text)
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)
