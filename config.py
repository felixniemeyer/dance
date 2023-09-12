# audio config

bitrate = 16
sample_rate = 44100

chunk_duration = 16
channels = 2
buffer_size = 512 # 86 FPS

# print info
print(f"bitrate: {bitrate}")
print(f"sample_rate: {sample_rate}")
print(f"chunk duration: {chunk_duration}")
print(f"channels: {channels}")
print(f"buffer_size: {buffer_size} => {int(sample_rate / buffer_size)} FPS")
print()
