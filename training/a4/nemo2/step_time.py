import re

def extract_steptime(log_file):
    count = 1
    eval_accuracies = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'"time_ms": ([\d.]+), "event_type": "POINT_IN_TIME", "key": "train_loss_update"', line)

            if match:
                #print(int(match.group(1))/1e3)
                count += 1
                eval_accuracies.append(int(match.group(1))/1e3)
    return eval_accuracies

import sys
log_file_path = sys.argv[1]
timestamps = extract_steptime(log_file_path)
prev_time = None
timesteps = []

for i in range(0, 10000, 40):
    for t in timestamps[i:i+40]:
        if prev_time is None:
            prev_time = t
        else:
            print(t - prev_time)
            timesteps.append(t - prev_time)
            prev_time = t
    prev_time = None
print(sum(timesteps) / len(timesteps))