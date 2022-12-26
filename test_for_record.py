import pyaudio
import wave
import os
import speech_recognition as sr

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 10
device_index = 2
audio = pyaudio.PyAudio()
print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
	if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
		print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
print("-------------------------------------------------------------")
index = int(input())		
print("recording via index "+str(index))
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,input_device_index = index,
                frames_per_buffer=CHUNK)
print ("recording started")
Recordframes = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
	data = stream.read(CHUNK,exception_on_overflow = False)
	Recordframes.append(data)
print("recording stopped")
stream.stop_stream()
stream.close()
audio.terminate()
OUTPUT_FILENAME="sample.wav"
# WAVE_OUTPUT_FILENAME=os.path.join("123",OUTPUT_FILENAME)
WAVE_OUTPUT_FILENAME = OUTPUT_FILENAME
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(Recordframes))
waveFile.close()

# from playsound import playsound
# playsound('sample.wav')

r = sr.Recognizer()
with sr.WavFile("sample.wav") as source:
	r.adjust_for_ambient_noise(source, duration = 0.5)
	audio = r.record(source)
try:
	#print(r.recognize_google(audio, show_all=True))
	print(r.recognize_google(audio))
except LookupError:
	print("Could not understand audio.")