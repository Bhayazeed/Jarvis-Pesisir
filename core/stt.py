"""
Speech-to-Text engines for MARK XL.

Whisper  – offline transcription via faster-whisper (VAD-buffered)
Vosk     – offline streaming transcription (lighter)
"""
import json
import numpy as np


class WhisperSTT:
    """Offline transcription using faster-whisper."""

    def __init__(self, model_name: str = "base", language: str | None = None):
        import os
        from faster_whisper import WhisperModel
        print(f"[STT] Loading Whisper '{model_name}'…")
        try:
            import torch
            device  = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
        except Exception:
            device, compute = "cpu", "int8"

        try:
            self._model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as _first_err:
            # Offline flag set but model not cached yet → clear flags and download once.
            # Keywords cover multiple huggingface_hub error message variants across versions.
            _e = str(_first_err).lower()
            _offline_keywords = (
                "offline", "not found", "cache", "localentry",
                "does not exist", "outgoing", "local_files_only",
            )
            if any(k in _e for k in _offline_keywords):
                print(f"[STT] Whisper '{model_name}' not in local cache — downloading (one-time, internet required)…")
                os.environ.pop("HF_HUB_OFFLINE",      None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                os.environ.pop("HF_DATASETS_OFFLINE",  None)
                try:
                    self._model = WhisperModel(model_name, device=device, compute_type=compute)
                except Exception as _dl_err:
                    raise RuntimeError(
                        f"Whisper '{model_name}' model download failed.\n"
                        f"Internet access is required the first time to download the speech model (~75–290 MB).\n"
                        f"After the first download it runs fully offline.\n"
                        f"Details: {_dl_err}"
                    ) from _dl_err
            else:
                raise

        self._language = None if (not language or language.strip().lower() == "auto") else language.strip().lower()
        print(f"[STT] Whisper '{model_name}' ready ({device})")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        try:
            segments, _ = self._model.transcribe(
                audio,
                language=self._language,
                beam_size=1,                       # greedy — 2-3x faster
                best_of=1,
                condition_on_previous_text=False,  # no hallucinations, faster
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            raise


class VoskSTT:
    """Streaming transcription using Vosk."""

    def __init__(self, model_path: str | None = None, language: str = "en-us"):
        from vosk import Model, KaldiRecognizer
        print("[STT] Loading Vosk model…")
        if model_path:
            model = Model(model_path)
        else:
            lang  = language.strip().lower() if language and language.strip().lower() != "auto" else "en-us"
            model = Model(lang=lang)
        self._rec = KaldiRecognizer(model, 16000)
        print("[STT] Vosk ready.")

    def process_chunk(self, audio_bytes: bytes) -> tuple[str, bool]:
        """Feed raw int16 LE PCM bytes. Returns (text, is_final)."""
        if self._rec.AcceptWaveform(audio_bytes):
            result = json.loads(self._rec.Result())
            return result.get("text", ""), True
        partial = json.loads(self._rec.PartialResult())
        return partial.get("partial", ""), False


class Wav2Vec2STT:
    """Offline transcription using a fine-tuned Wav2Vec 2.0 CTC model from Hugging Face."""

    def __init__(self, model_path: str, hf_token: str = None):
        import os
        import torch

        print(f"[STT] Loading Wav2Vec 2.0 from '{model_path}'…")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not hf_token and not os.path.exists(model_path):
            print("[STT] Warning: No HF token provided for potentially private model.")

        # Save existing offline environment variables
        old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
        old_trans_offline = os.environ.get("TRANSFORMERS_OFFLINE")

        # Temporarily disable offline mode to allow download if needed.
        # We must change env vars AND modify the cached huggingface_hub constants
        # + transformers.utils.is_offline_mode globals in case they were imported at startup.
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"

        try:
            import huggingface_hub.constants
            huggingface_hub.constants.HF_HUB_OFFLINE = False
        except ImportError:
            pass

        try:
            import transformers.utils
            if hasattr(transformers.utils, "is_offline_mode"):
                transformers.utils.is_offline_mode.__globals__["_is_offline_mode"] = False
        except Exception:
            pass

        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            self.processor = Wav2Vec2Processor.from_pretrained(
                model_path, token=hf_token, local_files_only=False,
            )
            model_loaded = Wav2Vec2ForCTC.from_pretrained(
                model_path, token=hf_token, local_files_only=False,
            ).to(self.device)
            
            if self.device == "cpu":
                try:
                    print("[STT] Optimizing Wav2Vec2 for CPU (applying dynamic int8 quantization)…")
                    self.model = torch.quantization.quantize_dynamic(
                        model_loaded, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("[STT] Dynamic int8 quantization applied successfully ✓")
                except Exception as qe:
                    print(f"[STT] Warning: Dynamic quantization failed: {qe}")
                    self.model = model_loaded
            else:
                self.model = model_loaded
        except Exception as e:
            raise RuntimeError(
                f"Wav2Vec '{model_path}' model download failed.\n"
                f"Internet access is required the first time to download the speech model.\n"
                f"After the first download it runs fully offline.\n"
                f"Details: {e}"
            ) from e
        finally:
            # Restore offline variables to maintain offline mode for subsequent operations
            if old_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = old_hf_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)

            if old_trans_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = old_trans_offline
            else:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)

            # Restore cached module constants
            try:
                import huggingface_hub.constants
                huggingface_hub.constants.HF_HUB_OFFLINE = (os.environ.get("HF_HUB_OFFLINE") == "1")
            except Exception:
                pass

            try:
                import transformers.utils
                if hasattr(transformers.utils, "is_offline_mode"):
                    transformers.utils.is_offline_mode.__globals__["_is_offline_mode"] = (
                        os.environ.get("TRANSFORMERS_OFFLINE") == "1"
                    )
            except Exception:
                pass

        print(f"[STT] Wav2Vec 2.0 Ready ({self.device})")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array. Returns transcript string."""
        import torch

        try:
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            return transcription.strip()
        except Exception as e:
            print(f"[STT] Wav2Vec transcription error: {e}")
            raise


class RemoteSTT:
    """STT client that sends audio to a remote REST API endpoint for transcription."""

    def __init__(self, api_url: str):
        print(f"[STT] Initializing Remote STT client with endpoint: {api_url}…")
        self.api_url = api_url

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 mono 16 kHz numpy array by calling the remote REST API."""
        import io
        import requests
        import soundfile as sf

        try:
            # 1. Convert float32 numpy array to WAV bytes in memory
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, 16000, format="WAV", subtype="PCM_16")
            wav_io.seek(0)

            # 2. POST the WAV bytes to the API
            files = {"file": ("audio.wav", wav_io, "audio/wav")}
            headers = {"accept": "application/json"}
            
            resp = requests.post(self.api_url, files=files, headers=headers, timeout=60)
            resp.raise_for_status()
            
            # 3. Parse and return the transcribed text
            data = resp.json()
            return (data.get("text") or "").strip()
            
        except Exception as e:
            print(f"[STT] Remote API transcription error: {e}")
            raise RuntimeError(f"Remote STT call failed: {e}")


