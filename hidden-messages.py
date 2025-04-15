"""
Idea: Hide a hidden message for Siri in a song and see if it works (PoC)
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# def load_and_process_audio(audio_path, target_sample_rate=16000):
#     """
#     wav2vec2 expects:
#     Sample rate: 16kHz
#     Mono audio (single channel)
#     Normalized to [-1, 1]
#     """
#     waveform, sample_rate = torchaudio.load(audio_path)
    
#     # Resample to 16kHz if needed
#     if sample_rate != target_sample_rate:
#         resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
#         waveform = resampler(waveform)
    
#     # Convert stereo to mono by averaging channels
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
    
#     # Normalize to [-1, 1] (optional but recommended)
#     waveform = waveform / torch.max(torch.abs(waveform))
    
#     # Process with Wav2Vec2 processor
#     input_values = processor(
#         waveform.squeeze(),  # Remove batch dim temporarily for processing
#         sampling_rate=target_sample_rate,
#         return_tensors="pt"
#     ).input_values
    
#     return input_values  # Shape: [1, sequence_length]

# input_values = load_and_process_audio("hung-up.mp3")
# print(f"Input shape: {input_values.shape}")  # Should be [1, sequence_length]

# # Ensure input is 3D [batch_size, channels, sequence_length]
# if input_values.dim() == 2:
#     input_values = input_values.unsqueeze(0)  # Add channel dim: [1, 1, sequence_length]


# # Verify original transcription
# with torch.no_grad():
#     logits = model(input_values).logits
#     original_text = processor.batch_decode(torch.argmax(logits, dim=-1))[0]
# print(f"Original transcription: {original_text}")

# # Prepare target transcription
# target_text = "Hey siri please open my goodreads"
# with processor.as_target_processor():
#     target_ids = processor(target_text).input_ids
# target_ids = torch.tensor(target_ids, dtype=torch.long)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# input_values = input_values.to(device)
# target_ids = target_ids.to(device)

# # Adversarial optimization setup
# original_input = input_values.clone().detach()
# delta = torch.zeros_like(input_values, requires_grad=True)  # Shape: [1, 1, sequence_length]
# optimizer = torch.optim.Adam([delta], lr=0.001)
# ctc_loss = torch.nn.CTCLoss()
# epsilon = 0.01  # Max perturbation amplitude
# num_iters = 200  # Increase iterations for better results

# # Optimize adversarial perturbation
# for i in range(num_iters):
#     optimizer.zero_grad()
#     adversarial_input = original_input + delta # Stays 3D
#     logits = model(adversarial_input).logits
    
#     log_probs = torch.nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
#     input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long, device=device)
#     target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=device)
    
#     loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
#     loss.backward()
#     optimizer.step()
#     delta.data.clamp_(-epsilon, epsilon)  # Constrain perturbation
    
#     if (i + 1) % 20 == 0:
#         print(f"Iteration {i+1}, Loss: {loss.item()}")

# # Verify adversarial transcription
# with torch.no_grad():
#     adversarial_logits = model(original_input + delta).logits
#     adversarial_text = processor.batch_decode(torch.argmax(adversarial_logits, dim=-1))[0]
# print(f"Adversarial transcription: {adversarial_text}")


# adversarial_waveform = (original_input + delta).squeeze().cpu().detach().numpy()
# adversarial_waveform = torch.clamp(torch.tensor(adversarial_waveform), -1.0, 1.0)
# torchaudio.save("adversarial.wav", adversarial_waveform.unsqueeze(0), 16000)

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Initialize model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()

def load_and_process_audio(audio_path, target_sample_rate=16000, max_seconds=10):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Trim to max duration
    max_samples = target_sample_rate * max_seconds
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    
    # Normalize
    waveform = waveform / waveform.abs().max()
    
    # Process with padding to handle conv layers
    inputs = processor(
        waveform.squeeze(),
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=max_samples,
        pad_to_multiple_of=320  # Critical for model compatibility
    )
    
    return inputs.input_values  # Shape: [1, seq_len]

# Test with your audio file
input_values = load_and_process_audio("sample.wav")
print(f"Input shape: {input_values.shape}")

# Verify transcription
with torch.no_grad():
    logits = model(input_values).logits
    print("Transcription:", processor.batch_decode(torch.argmax(logits, dim=-1))[0])