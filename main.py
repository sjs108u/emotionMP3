"""
    this program is used to detect user's mood and play some music for user.

    author: Chan-Sheng, Su
    contact: oracle1025@gmail.com
    department: Computer science and Information Engineering, Nationnal Chung Cheng University
"""

import Play_mp3
# utf-16 is not supported for some music files.
# Some content should be changed in C:\Users\User\anaconda3\Lib\site-packages\playsound.py (in which you download).
# See https://i.imgur.com/unYgRYo.png

import os
import random
import pyaudio
import wave
import speech_recognition as sr
import transformers
from transformers import AutoTokenizer,TFBertModel
import tensorflow as tf

print('Welcome to use the program.\nThis program is used to detect user\'s mood and play some music for user.')

while True:
    # input until correct syntax
    mode = input('Please enter the mode(0:random music 1:specified type of music 2:relative to mood 3:music on the contrary of mood):')
    # print(mode)
    if mode == '0' or mode == '1' or mode == '2' or mode == '3':
        break
    else:
        print('Incorrect input. Please try again.\n')

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
#bert = TFBertModel.from_pretrained('bert-base-cased')

encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
model2 = tf.keras.models.load_model('model/model_name.h5', custom_objects={"TFBertModel": transformers.TFBertModel})

if mode == '0': # random music played
    # Play_mp3.play('./test.mp3') # recommended
    # os.system('test.mp3') # will open another music player, not recommended
    dirs = os.listdir('music')
    
    while True:
        # anger, fear, joy, love, sadness, surprise  <-  6 moods 
        dir_name = dirs[random.randint(0, 6 - 1)]
        files = os.listdir(os.path.join('music', dir_name))

        count = 0
        for f in files:
            count = count + 1
        
        # eg, music/angry/test.mp3
        music = os.path.join('music', dir_name, files[random.randint(0, count - 1)])
        print(f'music:{music}. Enjoy!')
        Play_mp3.play(music)

elif mode == '1': # specified type of music
    while True:
        mytype = input('Here are some types of music for you!(0:anger 1:fear 2:joy 3:love 4:sadness 5:surprise):')
        if str.isdigit(mytype):
            mytype = int(mytype)
            break

    dirs = os.listdir('music')
    dir_name = dirs[mytype]
    files = os.listdir(os.path.join('music', dir_name))

    count = 0
    for f in files:
        count = count + 1
    
    # eg, music/angry/test.mp3
    while True:
        music = os.path.join('music', dir_name, files[random.randint(0, count - 1)])
        print(f'music:{music}. Enjoy!')
        Play_mp3.play(music)

elif mode == '2': # music relative to mood
    # Get input voice from user.
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
    
    while True:
        audio = pyaudio.PyAudio()
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

        # Change the sound to words with Google API(or others?).
        r = sr.Recognizer()
        with sr.WavFile("sample.wav") as source:
            r.adjust_for_ambient_noise(source, duration = 0.5)
            audio = r.record(source)
        try:
            #print(r.recognize_google(audio, show_all=True))
            print(r.recognize_google(audio))
            texts = r.recognize_google(audio)
        except LookupError:
            print("Could not understand audio.")

        # Predict the mood of the sound with our model.
        x_val = tokenizer(
            text=texts,
            add_special_tokens=True,
            max_length=70,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)

        validation = model2.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
        # print(validation)

        mood = 'a'
        max_acc = 0.0
        for key , value in zip(encoded_dict.keys(),validation[0]):
            print(key,value)
            if value > max_acc:
                max_acc = value
                mood = key
        print(mood, ':', max_acc)

        # Play music relative to the user's mood.

        dirs = os.listdir('music')
        # encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
        dir_name = dirs[encoded_dict[mood]]
        files = os.listdir(os.path.join('music', dir_name))
        
        count = 0
        for f in files:
            count = count + 1

        # eg, music/angry/test.mp3
        music = os.path.join('music', dir_name, files[random.randint(0, count - 1)])
        print(f'music:{music}. Enjoy!')
        Play_mp3.play(music)
    

elif mode == '3': # music on the contrary of mood
    # Get input voice from user.
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

    while True:
        audio = pyaudio.PyAudio()
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

        # Change the sound to words with Google API(or others?).
        r = sr.Recognizer()
        with sr.WavFile("sample.wav") as source:
            r.adjust_for_ambient_noise(source, duration = 0.5)
            audio = r.record(source)
        try:
            #print(r.recognize_google(audio, show_all=True))
            print(r.recognize_google(audio))
            texts = r.recognize_google(audio)
        except LookupError:
            print("Could not understand audio.")

        # Predict the mood of the sound with our model.
        x_val = tokenizer(
            text=texts,
            add_special_tokens=True,
            max_length=70,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True)

        validation = model2.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
        # print(validation)
        
        mood = 'a'
        max_acc = 0.0
        for key , value in zip(encoded_dict.keys(),validation[0]):
            print(key,value)
            if value > max_acc:
                max_acc = value
                mood = key
        
        print(mood, ':', max_acc)

        # Play music on the contraty of the user's mood.
        dirs = os.listdir('music')
        # encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}
        encoded_dict_inverse  = {'anger':3,'fear':2, 'joy':4, 'love':0, 'sadness':5, 'surprise':1}
        dir_name = dirs[encoded_dict_inverse[mood]]
        files = os.listdir(os.path.join('music', dir_name))
        
        count = 0
        for f in files:
            count = count + 1

        # eg, music/angry/test.mp3
        music = os.path.join('music', dir_name, files[random.randint(0, count - 1)])
        print(f'music:{music}. Enjoy!')
        Play_mp3.play(music)