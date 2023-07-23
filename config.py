bitrate = 16
sample_rate = 44100

chunk_duration = 16
channels = 2
buffer_size = 512 # 86 FPS

sequence = 16

size = bitrate * sample_rate * chunk_duration * channels

sequence_size = sample_rate * chunk_duration // buffer_size

sequence_offset = sample_rate * chunk_duration % buffer_size

# print info
print("config (edit config.py to make changes)")
print(f"bitrate: {bitrate}")
print(f"sample_rate: {sample_rate}")
print(f"chunk duration: {chunk_duration}")
print(f"channels: {channels}")
print(f"size: {size}")
print(f"buffers_per_file: {sequence_size}")
print(f"sequence_offset: {sequence_offset}")
print()
