from gtts import gTTS

tts = gTTS(text="xin chào tôi là Chiến", lang='vi')
tts.save('test.mp3')
print("convert successfully")