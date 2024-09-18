# Audio Processing and Transcription Pipeline

This repository provides a comprehensive Python-based pipeline for audio file processing, including conversion to WAV format, chunking large audio files, and performing speech-to-text transcription. The pipeline utilizes modern tools and libraries for efficient and accurate audio data handling.

## Features

- **Audio File Conversion**: Converts various audio file formats to WAV.
- **Audio Chunking**: Splits large audio files into smaller, manageable chunks.
- **Speech-to-Text Transcription**: Uses pre-trained models for transcribing audio chunks into text.
- **Integration with Hugging Face Hub**: Access and utilize pre-trained models for transcription.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/audio-processing-pipeline.git

2. Install the required packages:

    ```bash
    %pip install pydub soundfile librosa torch accelerate torchaudio datasets transformers huggingface_hub

## Usage

1. Convert Audio to WAV Format

    Replace input_file and output_directory with your file path and desired output directory.

    ```python
    import mimetypes
    import os
    from pydub import AudioSegment

    def is_audio_file(file_path):
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type and mime_type.startswith('audio')

    def convert_to_wav(file_path, output_dir):
        if not is_audio_file(file_path):
            print(f"{file_path} is not an audio file.")
            return

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.wav")

        try:
            audio = AudioSegment.from_file(file_path)
            audio.export(output_path, format='wav')
            print(f"File converted and saved as {output_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    input_file = "Kurdish Fairy Tales.mp3"
    output_directory = "./"
    convert_to_wav(input_file, output_directory)

2. Split Audio and Transcribe
    Replace input_file and output_directory with your file path and desired output directory.

    ```python
    from pydub import AudioSegment
    import os
    import librosa
    from transformers import Wav2Vec2ForCTC, AutoProcessor
    import torch

    def split_audio_to_chunks(file_path, output_dir, chunk_length_ms=60000):
        audioFull = AudioSegment.from_file(file_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        audio_length_ms = len(audioFull)
        final_transcription = ""

        for i in range(0, audio_length_ms, chunk_length_ms):
            start_ms = i
            end_ms = min(i + chunk_length_ms, audio_length_ms)

            chunk = audioFull[start_ms:end_ms]
            chunk_filename = os.path.join(output_dir, f"chunk_{i // chunk_length_ms + 1}.wav")
            chunk.export(chunk_filename, format="wav")
            print(f"Exported {chunk_filename}")

            audio_path = chunk_filename
            sampling_rate = 16000
            audio, _ = librosa.load(audio_path, sr=sampling_rate)

            processor = AutoProcessor.from_pretrained("facebook/mms-1b-fl102")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-fl102")
            processor.tokenizer.set_target_lang("gle")
            model.load_adapter("gle")

            inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs).logits

            ids = torch.argmax(outputs, dim=-1)[0]
            transcription = processor.decode(ids)
            print(transcription)
            final_transcription += transcription

        return final_transcription

    input_file = "President of Ireland speaking in Gaelic (St. Patrick s Day Message).mp3"
    output_directory = "tempAudio_chunks"
    transcription = split_audio_to_chunks(input_file, output_directory) 

## Dependencies
* pydub: For audio file conversion and manipulation.
* soundfile: For reading and writing audio files.
* librosa: For audio processing and analysis.
* torch: For working with PyTorch models.
* transformers: For utilizing pre-trained models from Hugging Face.
* huggingface_hub: For interacting with the Hugging Face Hub.

## License
This project is licensed under the MIT License - see the LICENSE file for details.