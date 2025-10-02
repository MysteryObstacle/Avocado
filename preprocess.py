import gzip
import os.path
from pathlib import Path

import click
import json

import numpy as np
from scapy.compat import raw
from scapy.layers.inet import TCP, UDP, IP
from scapy.layers.l2 import Ether

from utils import read_pcap, read_csv, packets_shunt, modbus_filter_func, modbus_label_func, sliding_window


# 步骤
# 1. 读取pcap文件
# 2. 过滤掉不需要的包
# 3. 流切割
# 4. 过滤掉连续数据包小于10的流
# 5. 包处理（去掉以太网头，udp填充，ip掩码）
# 6. 特征提取 [packet_count, packet_length]

def transform_pcap(path, output_path: Path = None, packet_count=10, max_length=245):
    """if Path(str(output_path.absolute()) + "_SUCCESS").exists():
        print(output_path, "Done")
        return"""
    # 流信息统计
    packets = read_pcap(path)
    flow_dict = packets_shunt(packets, modbus_filter_func, modbus_label_func, path, max_length)

    # 滑动窗口生成数据
    data_list = sliding_window(flow_dict, packet_count)

    print(path)
    # 写入文件
    if data_list:
        part_output_path = Path(
            str(output_path.absolute()) + f"_part_{0:04d}.json.gz"
        )
        with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
            for row in data_list:
                f_out.write(f"{json.dumps(row)}\n")

    # write success file
    with Path(str(output_path.absolute()) + "_SUCCESS").open("w") as f:
        f.write("")

    print(output_path, "Done")


def transform_csv(path, output_path: Path = None, packet_count=10, max_length=245):
    def transform(byte_data: bytes, max_length:int):
        arr = np.frombuffer(byte_data, dtype=np.uint8)[:max_length]
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

    def packets_shunt(packets, max_length=245, include_raw=False) -> dict:
        flow_dict = {}
        # 过滤无关的数据包
        for index, row in packets.iterrows():
            key = 'Modbus_' + str(row['Source']) + '_' + str(row['Destination']) + '_' + str(0)
            if key is None:
                continue
            if key not in flow_dict:
                flow_dict[key] = []

            label = row['Category']
            frame = row['Frame']

            try:
                byte_data = bytes.fromhex(frame)
            except ValueError as e:
                print(f"Error converting frame to bytes: {e}")
                continue

            if include_raw:
                value = (transform(byte_data, max_length).tolist(), label, byte_data)
            else:
                value = (transform(byte_data, max_length).tolist(), label)

            flow_dict[key].append(value)
            print("flow_dict is ok!")
        return flow_dict

    packets = read_csv(path)
    filtered_packets = packets[~(((packets['Source'] == 1) & (packets['Destination'] == 2)) |
                                 ((packets['Source'] == 3) & (packets['Destination'] == 2)))]

    flow_dict = packets_shunt(filtered_packets, max_length)
    data_list = sliding_window(flow_dict, packet_count, 7)

    if data_list:
        part_output_path = Path(
            str(output_path.absolute()) + f"_part_{0:04d}.json.gz"
        )
        with part_output_path.open("wb") as f, gzip.open(f, "wt") as f_out:
            for row in data_list:
                f_out.write(f"{json.dumps(row)}\n")

    # write success file
    with Path(str(output_path.absolute()) + "_SUCCESS").open("w") as f:
        f.write("")

def build_file_tree(root_dir):
    file_tree = {}
    for root, dirs, files in os.walk(root_dir):
        current_dict = file_tree
        current_path = root.replace(root_dir, '')
        folders = current_path.split(os.path.sep)

        for folder in folders:
            if folder:
                if folder not in current_dict:
                    current_dict[folder] = {}
                current_dict = current_dict[folder]

        for file in files:
            if file.endswith('.pcap') or file.endswith('.pcapng') or file.endswith('.csv'):
                current_dict[file] = None

    return file_tree


def deal_pcaps(root):
    def for_file_tree(file_tree, path='', cb_func=None, write_dir_root=None, read_dir_root=None):
        for k, v in file_tree.items():
            if type(v) is dict:
                for_file_tree(v, os.path.join(path, k), cb_func, write_dir_root, read_dir_root)
            elif v is None:
                cb_func(path, k, write_dir_root, read_dir_root)
                # print(os.path.join(path, k))

    def pcap_deal_cb(pacp_path, pcap_file, write_dir_root, read_dir_root):
        # print(pacp_path, pcap_file, write_dir_root, read_dir_root)
        target_dir_path = Path(os.path.join(write_dir_root, pacp_path))
        target_dir_path.mkdir(parents=True, exist_ok=True)

        transform_pcap(
            os.path.join(read_dir_root, pacp_path, pcap_file),
            Path(os.path.join(write_dir_root, pacp_path, pcap_file + ".transformed")),
            packet_count=10
        )

    pcap_dir = os.path.join(root, 'pcap')
    deal_dir = os.path.join(root, 'deal')
    file_tree = build_file_tree(os.path.join(root, 'pcap'))
    for_file_tree(file_tree, '', pcap_deal_cb, deal_dir, pcap_dir)


def deal_csvs(root):
    def for_file_tree(file_tree, path='', cb_func=None, write_dir_root=None, read_dir_root=None):
        for k, v in file_tree.items():
            if type(v) is dict:
                for_file_tree(v, os.path.join(path, k), cb_func, write_dir_root, read_dir_root)
            elif v is None:
                cb_func(path, k, write_dir_root, read_dir_root)

    def csv_deal_cb(csv_path, csv_file, write_dir_root, read_dir_root):
        target_dir_path = Path(os.path.join(write_dir_root, csv_path))
        target_dir_path.mkdir(parents=True, exist_ok=True)

        transform_csv(
            os.path.join(read_dir_root, csv_path, csv_file),
            Path(os.path.join(write_dir_root, csv_path, csv_file + ".transformed")),
            packet_count=10
        )

    csv_dir = os.path.join(root, 'csv')
    deal_dir = os.path.join(root, 'deal')
    file_tree = build_file_tree(os.path.join(root, 'csv'))
    for_file_tree(file_tree, '', csv_deal_cb, deal_dir, csv_dir)


@click.command()
@click.option(
    "-t",
    "--type",
    help="type of the files to be processed [pcap, csv]",
    default="pcap"
)
@click.option(
    "-r",
    "--root",
    help="root directory of the original files",
    required=True,
)
def main(type, root):
    if type == 'pcap':
        deal_pcaps(root)
    elif type == 'csv':
        deal_csvs(root)
    else:
        print('type error')
        return


if __name__ == "__main__":
    main()
