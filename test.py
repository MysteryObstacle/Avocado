value = list(range(37))

packet_count = 10
stride = 10
print(len(value))
for i in range(0, len(value) - packet_count + 1, stride):
    group = value[i:i + packet_count]
    print(i, group)
