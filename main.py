from pydoc_data.topics import topics
from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json
from docx import Document
from TextCategory import TextCategory
from io import BytesIO
import base64



app = Sanic(__name__)
CORS(app)
# define model here
tc = TextCategory()
@app.route('/')
async def test(request):
    return json({'hello': 'world'})

@app.route('/txt', methods=['POST'])
async def txt_reader(request):
    json_body = request.json
    file = json_body['file']
    file = base64.b64decode(file)
    file = BytesIO(file).read().decode('utf-8')
    return json({'response': file})

@app.route('/doc', methods=['POST'])
@app.route('/docx', methods=['POST'])
async def docx_reader(request):
    json_body = request.json
    file = json_body['file']
    file = base64.b64decode(file)
    doc = Document(BytesIO(file))
    res = {}
    paragraphs = [p.text for p in doc.paragraphs]
    for i, paragraph in enumerate(paragraphs):
        res[f'p{i}'] = paragraph
    # corpus = Tfidf.word_preprocessing(paragraphs)
    # topics = tc.get_topics(corpus)
    # topics = tc.get_topics(corpus)
    # res = {'response': topics}
    return json(res)


host = '127.0.0.1'
port = 8000
if __name__ == '__main__':
    app.run(host=host, port=port, auto_reload=True)