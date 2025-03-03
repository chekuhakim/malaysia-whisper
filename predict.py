from cog import BasePredictor, Input, Path
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import librosa
from datasets import Audio

class Predictor(BasePredictor):
    def setup(self):
        self.processor = AutoProcessor.from_pretrained("mesolitica/malaysian-whisper-large-v3-turbo")
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained("mesolitica/malaysian-whisper-large-v3-turbo")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, audio: Path = Input(description="Audio file to transcribe")) -> str:
        audio_data, sr = librosa.load(str(audio), sr=16000)
        audio_dataset = Audio(sampling_rate=16000)
        processed_audio = audio_dataset.decode_example(audio_dataset.encode_example({"array": audio_data, "sampling_rate": 16000}))["array"]
        inputs = self.processor(processed_audio, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(inputs["input_features"], language="ms", return_timestamps=True)
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
