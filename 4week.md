```
4주차 인공지능 과제
```

[Audio_-_Free_practice_test_-_Listening_-_Matching_-_Sample_1.mp3](https://github.com/user-attachments/files/22630154/Audio_-_Free_practice_test_-_Listening_-_Matching_-_Sample_1.mp3)
```
Sample file
```
```
import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

SPEECH_FILE = "Audio_-_Free_practice_test_-_Listening_-_Matching_-_Sample_1.mp3"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

print("Sample Rate:", bundle.sample_rate)

print("Labels:", bundle.get_labels())

```
<img width="1406" height="108" alt="image" src="https://github.com/user-attachments/assets/326467c0-e3eb-4bcc-8d35-a653ce361f8d" />

```
model = bundle.get_model().to(device)

print(model.__class__)

<class 'torchaudio.models.wav2vec2.model.Wav2Vec2Model'>

```

```
fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest")
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")
fig.tight_layout()
```
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/6d9e12cf-6dc6-4534-a3f9-1c04bd1d50da" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/b0236c3a-4999-4765-8c18-6e110007fe10" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/ee29eab2-efaf-4b63-941b-a4e28a074767" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/4755d264-d7b6-45e3-9664-63c428538e05" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/3b1543ef-fd0d-4487-bc4b-34dd6becf348" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/36a2e28d-0da3-4d44-b2ce-ac229d533252" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/dc675c9f-4435-4531-97c4-30dfcd575a82" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/e6688254-e55e-4f22-a53d-761e8185df55" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/2267cfb4-550d-4ed8-b733-95d5f22a56ed" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/8f869c59-7090-4a70-bad7-eaf6cae0a5d5" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/05d60402-e090-43cd-94fb-f43b01e308ae" />
<img width="250" height="350" alt="image" src="https://github.com/user-attachments/assets/3dbe10a1-6dd5-4b94-aa51-d0426404f460" />


```
with torch.inference_mode():
    emission, _ = model(waveform)
plt.imshow(emission[0].cpu().T, interpolation="nearest")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
print("Class labels:", bundle.get_labels())

```
<img width="1421" height="207" alt="image" src="https://github.com/user-attachments/assets/1e575679-3bf3-4838-a129-b0459224649d" />

```
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

print(transcript)
IPython.display.Audio(SPEECH_FILE)
```

<img width="1466" height="852" alt="image" src="https://github.com/user-attachments/assets/9831a3e9-bdc9-4987-a512-3583604e1364" />
