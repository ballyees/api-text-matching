from sanic import Sanic
from sanic_cors import CORS
from sanic.response import json
from docx import Document
from TextCategory import TextCategoryInstance, Preprocessor
from io import BytesIO
import numpy as np
import base64


app = Sanic(__name__)
CORS(app)
# define model here
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
    
    paragraphs = [p.text for p in doc.paragraphs]
    # res = {}
    # for i, paragraph in enumerate(paragraphs):
    #     res[f'p{i}'] = paragraph
    # corpus = Tfidf.word_preprocessing(paragraphs)
    p = Preprocessor()
    corpus = p.word_preprocessing(paragraphs)
    topics = TextCategoryInstance.get_topics(corpus)
    topics_unique = np.hstack([t.split(', ') for t in topics])
    topics_unique = np.unique(topics_unique)
    res = {'response': topics, 'unique': topics_unique.tolist()}
    return json(res)


host = '127.0.0.1'
port = 8000
if __name__ == '__main__':
    try:
        app.run(host=host, port=port, auto_reload=False, workers=10)
    except KeyboardInterrupt:
        exit(1)
        # cmd command for kill all python process
        # "taskkill /f /im "python.exe" /t"