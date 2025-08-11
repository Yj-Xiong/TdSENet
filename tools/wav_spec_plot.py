import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 加载音频文件
audio_path = 'your_audio_file.wav'
y, sr = librosa.load(audio_path, sr=16000)  # sr=16000 将采样率设置为 16000 Hz

# 绘制波形图
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# 计算STFT
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 绘制STFT频谱
plt.figure(figsize=(14, 5))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT Magnitude Spectrum')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()
