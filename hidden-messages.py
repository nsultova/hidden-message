"""
Idea: Hide a hidden message for Siri in a song and see if it works (PoC)
"""
# import torch
# import torchaudio
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()

# def load_and_process_audio(audio_path, target_sample_rate=16000, max_seconds=10):
#     waveform, sample_rate = torchaudio.load(audio_path)
    
#     if sample_rate != target_sample_rate:
#         waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    
#     # Convert to mono, most speec models trained on mono encoding, one channel
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
    
#     # Trim to max duration
#     max_samples = target_sample_rate * max_seconds
#     if waveform.shape[1] > max_samples:
#         waveform = waveform[:, :max_samples]
    
#     # Normalize
#     waveform = waveform / waveform.abs().max()
    
#     # Paddding for conv layers
#     inputs = processor(
#         waveform.squeeze(),
#         sampling_rate=target_sample_rate,
#         return_tensors="pt",
#         padding="max_length",
#         max_length=max_samples,
#         pad_to_multiple_of=320  # Make audio length multiple of 320 (Wav2Vec conv stride pattern)
    
#     return inputs.input_values  # Shape: [1, seq_len]


# input_values = load_and_process_audio("sample.wav")
# print(f"Input shape: {input_values.shape}")

# # Verify transcription
# with torch.no_grad():
#     logits = model(input_values).logits
#     print("Transcription:", processor.batch_decode(torch.argmax(logits, dim=-1))[0])

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

# Configuration
TARGET_PHRASE = "hey siri please open my goodreads"
SEGMENT_START = 30  # Start at 30 seconds into the song
SEGMENT_DURATION = 5  # Work with 5-second segment
EPSILON = 0.03  # Max perturbation (keep under 0.05 for imperceptibility)
NUM_ITERS = 500  # Optimization iterations

def load_and_process_segment(audio_path, start_sec, duration_sec):
    # Load full audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono and resample to 16kHz if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    
    # Extract target segment
    start_sample = int(start_sec * 16000)
    end_sample = start_sample + int(duration_sec * 16000)
    segment = waveform[:, start_sample:end_sample]
    
    # Normalize and prepare
    segment = segment / torch.max(torch.abs(segment))
    inputs = processor(
        segment.squeeze(),
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        max_length=int(duration_sec * 16000),
        pad_to_multiple_of=320
    )
    return inputs.input_values, waveform, start_sample, end_sample

# Initialize model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load audio segment
input_values, original_waveform, start, end = load_and_process_segment(
    "hung-up.mp3", SEGMENT_START, SEGMENT_DURATION
)
original_segment = input_values.clone().to(device)

# Prepare target phrase properly
target_ids = processor.tokenizer(
    TARGET_PHRASE,
    return_tensors="pt",
    padding="max_length",
    max_length=64,
    truncation=True
).input_ids.squeeze()

# Filter out special tokens (keep only actual text tokens)
target_ids = target_ids[(target_ids != processor.tokenizer.pad_token_id) &
                        (target_ids != processor.tokenizer.bos_token_id) &
                        (target_ids != processor.tokenizer.eos_token_id)]

target_ids = target_ids.to(device)

# Adversarial optimization
delta = torch.zeros_like(original_segment, requires_grad=True)
optimizer = torch.optim.Adam([delta], lr=0.001)
ctc_loss = torch.nn.CTCLoss()

for i in range(NUM_ITERS):
    optimizer.zero_grad()
    adv_input = original_segment + delta
    logits = model(adv_input).logits
    
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)
    input_lengths = torch.tensor([logits.shape[1]], device=device)
    target_lengths = torch.tensor([len(target_ids)], device=device)
    
    loss = ctc_loss(log_probs, target_ids, input_lengths, target_lengths)
    loss.backward()
    optimizer.step()
    delta.data.clamp_(-EPSILON, EPSILON)
    
    if (i+1) % 50 == 0:
        print(f"Iter {i+1}: Loss={loss.item():.3f}")

# Insert adversarial segment back into original audio
perturbed_segment = (original_segment + delta).cpu().squeeze()
perturbed_segment = torch.clamp(perturbed_segment, -1.0, 1.0)

# Maintain original audio quality by only modifying high-frequency components
original_segment_np = original_waveform[0, start:end].numpy()
perturbed_segment_np = perturbed_segment.numpy()

# Blend using spectral masking (preserves audio quality)
def spectral_blend(original, perturbed, alpha=0.9):
    fft_orig = np.fft.rfft(original)
    fft_pert = np.fft.rfft(perturbed)
    
    # Keep original low frequencies, add perturbed high frequencies
    cutoff = int(len(fft_orig) * alpha)
    blended = np.concatenate([
        fft_orig[:cutoff],
        fft_pert[cutoff:]
    ])
    return np.fft.irfft(blended).astype(np.float32)

final_segment = spectral_blend(original_segment_np, perturbed_segment_np)

# Replace segment in original audio
final_waveform = original_waveform.clone()
final_waveform[0, start:end] = torch.from_numpy(final_segment)

# Save result
torchaudio.save(
    "hidden_command_song.wav",
    final_waveform,
    16000,
    format="wav",
    bits_per_sample=16
)

print("Adversarial song saved! Test with:") 
print(f"1. Play hidden_command_song.wav from {SEGMENT_START}s mark")
print(f"2. Check transcription with Siri/voice assistants")
print(f"3. Human ears should hear normal music at {SEGMENT_START}s")