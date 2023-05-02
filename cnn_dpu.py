import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Define constants
TIME_WINDOW = 5
MAX_PACKETS = 100
THRESHOLD = 0.5

# Initialize labeled dataset
dataset = []

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(MAX_PACKETS, 11)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Capture and process live network traffic
while True:
    # Get the next packet
    packet = get_next_packet()
   
    # If packet capture time falls within the time window
    if packet.capture_time >= dataset[-1][0] + TIME_WINDOW:
        # Group packets by bi-directional traffic flow
        flows = group_packets_by_flow(packet)
        for flow in flows:
            # Truncate or pad flows to max length
            if len(flow) > MAX_PACKETS:
                flow = flow[:MAX_PACKETS]
            else:
                flow = np.pad(flow, ((0, MAX_PACKETS - len(flow)), (0, 0)), 'constant')
            # Normalize flow data
            flow = (flow - np.mean(flow, axis=0)) / np.std(flow, axis=0)
            # Add flow to the labeled dataset
            dataset.append((packet.capture_time, flow, packet.label))
       
        # Train the CNN model on the labeled dataset
        X = np.array([f[1] for f in dataset])
        y = np.array([f[2] for f in dataset])
        model.fit(X, y, epochs=1, batch_size=32, verbose=0)
       
    # Classify the new traffic flow
    new_flow = get_new_traffic_flow()
    new_flow = (new_flow - np.mean(new_flow, axis=0)) / np.std(new_flow, axis=0)
    new_flow = np.expand_dims(new_flow, axis=0)
    prediction = model.predict(new_flow)
   
    # Take appropriate mitigation actions if the flow is an attack above the threshold
    if prediction > THRESHOLD:
        if get_flow_severity(new_flow) > THRESHOLD:
            take_mitigation_actions(new_flow)


=================================================
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from scapy.all import *

# Define constants
TIME_WINDOW = 5  # seconds
MAX_PACKETS = 100
THRESHOLD = 0.9

# Initialize labeled dataset
dataset = []

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(MAX_PACKETS, 11)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def get_next_packet():
    """
    Sniff and return the next packet from the network interface.
    """
    packet = sniff(count=1)[0]
    return packet

def group_packets_by_flow(packet):
    """
    Group packets by bi-directional traffic flow.
    """
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    src_port = packet.sport
    dst_port = packet.dport
    flow1 = []
    flow2 = []
    for pkt in dataset[::-1]:
        if pkt[0] < packet.time - TIME_WINDOW:
            break
        if pkt[1][0][1] == src_ip and pkt[1][0][2] == dst_ip and pkt[1][0][4] == src_port and pkt[1][0][5] == dst_port:
            flow1.append(pkt[1])
        elif pkt[1][0][1] == dst_ip and pkt[1][0][2] == src_ip and pkt[1][0][4] == dst_port and pkt[1][0][5] == src_port:
            flow2.append(pkt[1])
    flows = flow1 + flow2[::-1]
    return flows

def get_new_traffic_flow():
    """
    Sniff and return a new traffic flow from the network interface.
    """
    flow = []
    while len(flow) < MAX_PACKETS:
        packet = get_next_packet()
        if TCP in packet:
            flow.append((len(packet), packet[IP].src, packet[IP].dst, packet[IP].proto, packet[TCP].sport, packet[TCP].dport, packet.time, packet[TCP].flags, packet[IP].ttl, packet[IP].version, packet[IP].ihl, packet[IP].tos, packet[IP].flags, packet[IP].frag))
    flow = np.array(flow, dtype=np.float32)
    return flow

def get_flow_severity(flow):
    """
    Calculate the severity of the traffic flow.
    """
    X = np.expand_dims(flow, axis=0)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    severity = model.predict(X)
    return severity

def take_mitigation_actions(flow):
    """
    Take appropriate mitigation actions for the traffic flow.
    """
    print(f"DDoS attack detected with severity {get_flow_severity(flow)}")
    # drop packets, reroute traffic, or notify administrator

# Capture and process live network traffic
while True:
    # Get the next packet
    packet = get_next_packet()

    # If packet capture time falls within the time window
    if packet.time >= dataset[-1][0] + TIME_WINDOW:
        # Group packets by bi-direction
