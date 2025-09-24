```
## WAVEFORM 출력

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")

```
<img width="1536" height="833" alt="image" src="https://github.com/user-attachments/assets/9fe2f2a6-59c0-4658-8c74-6ced027fac9c" />

```
## Spectrogram 출력

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
```
<img width="1577" height="833" alt="image" src="https://github.com/user-attachments/assets/cdf18d87-7d83-43f2-94aa-10f85b0e32c5" />

```
## 오리지날 스펙트럼 출력

sample_rate = 48000
waveform = get_sine_sweep(sample_rate)

plot_sweep(waveform, sample_rate, title="Original Waveform")
Audio(waveform.numpy()[0], rate=sample_rate)
```
<img width="1530" height="822" alt="image" src="https://github.com/user-attachments/assets/7fec7171-209c-47c0-9d5b-3d4a83034aff" />

```
## 다운샘플링 스펙트럼 출력

resample_rate = 32000
resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)

plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
Audio(resampled_waveform.numpy()[0], rate=resample_rate)
```
<img width="1524" height="825" alt="image" src="https://github.com/user-attachments/assets/b35c2fe6-1dbb-4cac-9721-a578fa899734" />

```
## 업샘플링 스펙트럼 출력

resample_rate = 64000
resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)

plot_sweep(resampled_waveform, resample_rate, title="Resampled Waveform")
Audio(resampled_waveform.numpy()[0], rate=resample_rate)
```
<img width="1506" height="843" alt="image" src="https://github.com/user-attachments/assets/bec6e832-57c6-437c-b8c2-4046279a5db3" />

