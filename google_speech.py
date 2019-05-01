import argparse
import io
import json
# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types


def transcribe(args):
    # Instantiates a client
    client = speech.SpeechClient()

    # Loads the audio into memory
    with io.open(args.file, 'rb') as audio_file:
        content = audio_file.read()

    if args.stream:
        stream = [content]
        requests = (types.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in stream)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=args.sample_rate_hertz,
            max_alternatives=args.num_speakers,
            model='video',
            language_code='en-US')
        streaming_config = types.StreamingRecognitionConfig(config=config)

        # streaming_recognize returns a generator.
        responses = client.streaming_recognize(streaming_config, requests)

        return responses[0].results
    else:
        audio = types.RecognitionAudio(content=content)

        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=args.sample_rate_hertz,
            max_alternatives=args.num_speakers,
            model='video',
            language_code='en-US')

        # Detects speech in the audio file
        response = client.recognize(config, audio)

        return response.results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-n', '--num-speakers', type=int, default=1)
    parser.add_argument('-s', '--stream', action='store_true')
    parser.add_argument('--sample-rate-hertz', type=int)
    args = parser.parse_args()

    results = transcribe(args)
    for result in results:
        print(json.dumps(dict(transcript=result.alternatives[0].transcript.strip(),
                              confidence=result.alternatives[0].confidence),
                         ensure_ascii=False))


if __name__ == '__main__':
    main()
