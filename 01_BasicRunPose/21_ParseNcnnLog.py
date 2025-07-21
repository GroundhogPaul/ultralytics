import re
from collections import defaultdict

def parse_log_file(log_file_path):
    layer_times = defaultdict(list)     # key: layer name; # value: time in ms
    
    # re pattern: (format：layername + ' ' + time + "ms")
    pattern = re.compile(r'(\S+)\s+(\d+\.\d+)ms')
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # match re pattern
            match = pattern.search(line)
            if match:
                layer_name = match.group(1)
                time_ms = float(match.group(2))
                layer_times[layer_name].append(time_ms)
    
    # calculate average time for each layer
    results = {}
    for layer, times in layer_times.items():
        avg_time = sum(times) / len(times)
        results[layer] = avg_time
    
    return results

if __name__ == "__main__":
    log_path = "01_BasicRunPose/log.txt"
    avg_times = parse_log_file(log_path)
    
    # print result（most time spent）
    print("Layer Name\tAverage Time (ms)")
    print("-" * 35)
    for layer, time in sorted(avg_times.items(), key=lambda x: x[1], reverse=True):
        print(f"{layer}\t{time:.4f}")

    # total average time
    total_avg_time = sum(avg_times.values())
    avg_per_layer = total_avg_time / len(avg_times)
    
    print("\n{:<40} {:<15}".format("Total Average Time (ms)", "Average per Layer (ms)"))
    print("-" * 55)
    print("{:<40} {:<15.4f}".format(total_avg_time, avg_per_layer))