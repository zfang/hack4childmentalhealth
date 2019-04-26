import argparse

from sanic_chat_app import app

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1)
args = parser.parse_args()
app.run(workers=args.workers)
