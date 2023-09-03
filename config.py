bitrate = 16
sample_rate = 44100

chunk_duration = 16
channels = 2
buffer_size = 512 # 86 FPS

sequence = 16

teacher_forcing_size = chunk_duration * sample_rate // 512 // 3

# print info
print(f"bitrate: {bitrate}")
print(f"sample_rate: {sample_rate}")
print(f"chunk duration: {chunk_duration}")
print(f"channels: {channels}")
print(f"teacher_forcing_size: {channels}")
print()
