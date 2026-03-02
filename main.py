
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSequenceClassification

class AudioGrammarEngine:
    def __init__(self):
        # 1. Load Transcription Model (Wav2Vec2 - lightweight & fast)
        print("Loading Acoustic Model (Wav2Vec2)...")
        self.stt_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.stt_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

        # 2. Load Grammar Scoring Model (RoBERTa fine-tuned on CoLA)
        print("Loading Grammar Scorer (RoBERTa)...")
        self.score_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        self.score_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")

    def transcribe(self, audio_path):
        # Wav2Vec2 requires 16000Hz mono audio
        speech, rate = librosa.load(audio_path, sr=16000)
        
        inputs = self.stt_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.stt_model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = self.stt_processor.batch_decode(predicted_ids)[0]
        return transcript.lower()

    def score_grammar(self, text):
        if not text.strip(): return 0.0
        
        inputs = self.score_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.score_model(**inputs)
            # Index 1 corresponds to "grammatically acceptable"
            probs = torch.softmax(outputs.logits, dim=1)
            score = probs[0][1].item() * 100
        return round(score, 2)

    def evaluate_user_audio(self, audio_path):
        print(f"\nAnalyzing: {audio_path}")
        text = self.transcribe(audio_path)
        score = self.score_grammar(text)
        
        print("-" * 30)
        print(f"Transcript: {text}")
        print(f"Grammar Score: {score}/100")
        print("-" * 30)
        return {"transcript": text, "score": score}
engine = AudioGrammarEngine()
for i in range(1,4):
  result = engine.evaluate_user_audio("dataset/audio_"+str(i)+".wav")


# --- Execution ---
