# Terminology: 
# chunk = a piece of audio (e.g. 16 seconds long)
# buffer = equally sized chunks of audio
# sequence = a sequence of buffers

# audio config
bitrate = 16
samplerate = 44100

chunk_duration = 16
channels = 2
buffer_size = 512 # 86 FPS

# print info
print(f"bitrate: {bitrate}")
print(f"sample_rate: {samplerate}")
print(f"chunk duration: {chunk_duration}")
print(f"channels: {channels}")
print(f"buffer_size: {buffer_size} => {int(samplerate / buffer_size)} FPS")
print()
