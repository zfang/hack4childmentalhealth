import os
import ujson as json

import bleach
from sanic import Sanic
from sanic.log import logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SANIC_HOST = os.environ.get('SANIC_HOST', '127.0.0.1')
SANIC_PORT = os.environ.get('SANIC_PORT', 8000)
SANIC_DEBUG = os.environ.get('SANIC_DEBUG', False)

CONNECTIONS = {}

ERROR_NICKNAME_MISSING = json.dumps({'error': 'nickname is required'})
ERROR_NICKNAME_IN_USE = json.dumps({'error': 'nickname already in use'})

app = Sanic()

# Serving static files
app.static('/static', os.path.join(BASE_DIR, 'static/'))

# Serving index page
app.static('/', os.path.join(BASE_DIR, 'templates/index.html'), name='index')


@app.websocket('/chat')
async def chat(request, ws):
    nickname = request.args.get('nickname')
    if not nickname:
        return await ws.send(ERROR_NICKNAME_MISSING)

    if nickname in CONNECTIONS:
        return await ws.send(ERROR_NICKNAME_IN_USE)

    ws.nickname = nickname
    CONNECTIONS[nickname] = ws
    logger.info("{} has been connected!".format(nickname))

    try:
        while True:
            msg = await ws.recv()
            msg = json.dumps({"nickname": nickname, "message": bleach.linkify(bleach.clean(msg))})
            for user in CONNECTIONS.values():
                await user.send(msg)
    finally:
        del CONNECTIONS[nickname]
        logger.info("{} has been disconnected!".format(nickname))


def run(workers=1):
    app.run(host=SANIC_HOST, port=SANIC_PORT, debug=SANIC_DEBUG, workers=workers)


if __name__ == '__main__':
    run()
