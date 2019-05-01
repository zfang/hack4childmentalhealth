import argparse
import os
from pydub import AudioSegment
from tqdm import tqdm


def split(args, tqdm_func=tqdm):
    name, ext = os.path.splitext(args.file)
    if ext == '.wav':
        audio = AudioSegment.from_wav(args.file)
    else:
        raise NotImplementedError(args.ext)
    batch = args.batch * 1000
    for k, i in enumerate(tqdm_func(range(0, len(audio), batch))):
        audio[i:i + batch].export('{}_{}{}'.format(name, k, ext), format=ext[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-b', '--batch', type=int, default=60)
    args = parser.parse_args()
    split(args)


if __name__ == '__main__':
    main()
