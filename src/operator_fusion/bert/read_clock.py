f = open("/home/xiachunwei/Projects/tensor-compiler-gpu/release/block0.txt", 'r')
lines = f.readlines()
idx = 0
last_timestep = 0
for line in lines:
  line = line.strip()
  if idx == 0:
    last_timestep = int(line)
    idx+=1
  else:
    current = int(line)
    if(current==0):
      break;
    latency = (current - last_timestep) / 1410
    print(latency)
    last_timestep = current

