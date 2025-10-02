from pathlib import Path

import pandas as pd
from scapy.all import *
from scapy.layers.dns import DNS
from scapy.layers.inet import TCP, IP, UDP
from scapy.layers.l2 import Ether
import numpy as np


def get_umas_code(packet):
    if TCP in packet:
        if packet[TCP].dport == 502:
            modbus_hex_str = packet[TCP].payload
            return bytes(modbus_hex_str).hex()[18:20]

    return None


def read_pcap(path: Path):
    packets = PcapReader(str(path))
    return packets


def read_csv(path: Path):
    column_names = ['Frame', 'Category', 'Specific', 'Source', 'Destination', 'Time']
    packets = pd.read_csv(path, header=None, names=column_names)
    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True

    # DNS segment
    if DNS in packet:
        return True

    return False


def remove_ether_header(packet):
    if Ether in packet:
        packet = packet[Ether].payload


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"


def mask_port(packet):
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0
    elif TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0


def packet_to_sparse_array(packet, max_length=1500):
    # 转换原始字节数据并填充
    raw_packet = raw(packet[TCP].payload)
    arr = np.frombuffer(raw_packet, dtype=np.uint8)[:max_length]
    raw_bytes_padded = np.zeros(max_length, dtype=np.float32)
    raw_bytes_padded[:len(arr)] = arr

    # 生成1~max_length位置编码
    pos_enc = np.arange(0, max_length, 1, dtype=np.uint8)

    # 生成有无字节编码
    is_nonzero_encoding = np.zeros(max_length, dtype=np.uint8)
    is_nonzero_encoding[:len(arr)] = 1.0

    # 组合编码
    combined_encoding = np.stack((raw_bytes_padded, pos_enc, is_nonzero_encoding), axis=-1)

    return combined_encoding


def transform_packet(packet, max_length=1500):
    mask_ip(packet)
    mask_port(packet)
    remove_ether_header(packet)
    arr = packet_to_sparse_array(packet, max_length)

    return arr


def modbus_filter_func(packet):
    key = None

    # 如果是默认过滤的包(DNS, SYN, ACK, FIN...)
    if should_omit_packet(packet):
        return key

    # 如果是modbus request包
    if TCP in packet and packet[TCP].dport == 502:
        key = 'Modbus_' + packet[IP].src + '_' + packet[IP].dst + '_' + str(packet[TCP].sport)

    return key


def modbus_label_func(packet, path=None):
    label = 0

    umas_code_str = get_umas_code(packet)
    if umas_code_str:
        if path.find('traditional_attack') != -1 and umas_code_str == '29':
            label = 1
        elif path.find('dataExecution') != -1 and umas_code_str == '29':
            label = 2

    return label


def packets_shunt(packets, filter_func, label_func=None, path=None, max_length=245, include_raw=False) -> dict:
    """
    Args:
        packets: scapy.plist.PacketList
        filter_func: (packet: scapy.packet) -> str | None
        label_func: (packet: scapy.packet, path: Path) -> int | None
    """
    flow_dict = {}

    for packet in packets:
        key = filter_func(packet)
        if key is None:
            continue
        if key not in flow_dict:
            flow_dict[key] = []

        label = None
        if label_func:
            label = label_func(packet, path)

        if include_raw:
            value = (transform_packet(packet, max_length).tolist(), label, raw(packet[TCP].payload))
        else:
            value = (transform_packet(packet, max_length).tolist(), label)

        flow_dict[key].append(value)

    return flow_dict


def sliding_window(flow_dict, packet_count, stride=1, include_raw=False) -> list:
    global raws
    data_list = []

    for key, value in flow_dict.items():
        if len(value) < packet_count:
            continue

        # 遍历 flow_dict 中的每个流，以滑动窗口方式提取数据
        for i in range(0, len(value) - packet_count + 1, stride):
            if i + packet_count > len(value):
                break
            group = value[i:i + packet_count]

            if include_raw:
                raws = [item[2] for item in group]
            labels = [item[1] for item in group]
            features = [item[0] for item in group]

            data = {
                "attack_label": labels,
                "feature": features,
            }
            if include_raw:
                data["raws"] = raws

            data_list.append(data)

    return data_list
