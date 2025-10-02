import importlib
import os.path

import click
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
import datasets


from utils import read_pcap, packets_shunt, modbus_filter_func, modbus_label_func, sliding_window


# 示例模型加载（假设你有一个训练好的PyTorch模型）
def load_model(model_package, weights_path, gpu=True):
    if gpu:
        device = "cuda"
    else:
        device = "cpu"

    try:
        model_module = importlib.import_module(f"ml.{model_package}.{model_package}")
        model_class = getattr(model_module, model_package.capitalize())
    except (ImportError, AttributeError) as e:
        print(f"Error loading model package '{model_package}': {e}")
        return None

    model = (
        model_class.load_from_checkpoint(
            str(Path(weights_path).absolute()), map_location=torch.device(device)
        )
        .float()
        .to(device)
    )

    model.eval()

    return model


def transform_pcap(path, packet_count=10, max_length=245):
    # 流信息统计
    packets = read_pcap(path)
    flow_dict = packets_shunt(packets, modbus_filter_func, modbus_label_func, path=path, max_length=max_length, include_raw=True)

    # 滑动窗口生成数据
    data_list = sliding_window(flow_dict, packet_count, stride=packet_count, include_raw=True)

    return data_list


def transform_parquet(path, packet_count=10, max_length=245):
    dataset_dict = datasets.load_dataset("parquet", data_files=[path])
    dataset = dataset_dict[list(dataset_dict.keys())[0]]

    data_list = []

    for data in dataset:
        if "raws" not in data:
            data["raws"] = []
            for packet in data["feature"]:
                bytes = b""
                # 将value[0]，其为float类型，挨个取出来，转换成拼接起来，要保证是byte类型
                for value in packet:
                    bytes += int(value[0]).to_bytes(1, byteorder='big')
                data["raws"].append(bytes)

        transformed_batch = {"feature": data["feature"], "attack_label": data["label"], "raws": data["raws"]}
        data_list.append(transformed_batch)

    return data_list


def infer_model(model, data, packet_count=10, num_class=3, dimension=0):
    feature = torch.stack([torch.tensor(d["feature"]) for d in data])
    label = torch.tensor([d["attack_label"] for d in data])
    transformed_data = {"feature": feature, "label": label}


    with torch.no_grad():
        x = transformed_data["feature"].float().to(model.device)
        y = transformed_data["label"].reshape(-1).long()

        logits, weights = model(x)

        y_hat = torch.argmax(F.log_softmax(logits.reshape(-1, num_class), dim=1), dim=1)

    batch_size, packet_count, channel = logits.shape

    if dimension == -1:
        weights = weights.squeeze(2)
        weights = weights.reshape(batch_size, packet_count, -1)

    print(logits.shape)
    print(weights.shape)

    for index, value in enumerate(data):
        value['predicts'] = torch.argmax(F.log_softmax(logits[index], dim=1), dim=1).cpu().numpy()
        if dimension == -1:
            # (batch_size, input_dim, packet_count, packet_length)
            # (batch_size * packet_count, packet_length, 1)
            value['weights'] = weights[index].cpu().numpy()
        else:
            value['weights'] = weights[index][dimension].cpu().numpy() * 100

    accuracy = (y_hat.cpu() == y).float().mean().item()
    # 打印 y_hat != y 的index值，格式为 index, true, predict
    mismatched_indices = torch.where(y_hat.cpu() != y)[0]
    print("index, true, predict")
    for idx in mismatched_indices:
        print(f"{idx.item()}, {y[idx].item()}, {y_hat[idx].item()}")

    return data, accuracy


def generate_heatmap(results, output_path, start=60, end=120, packet_length=64, attention_threshold=0.065):
    data = []
    text_data = []
    predicts = []
    true_labels = []

    for r in results:
        for p in r['predicts']:
            predicts.append(p)
        for l in r['attack_label']:
            true_labels.append(l)
        for w in r['weights']:
            data.append(w[:packet_length])
        for raw in r['raws']:
            text_list = []
            for byte in raw[:packet_length]:
                hex_byte = format(byte, '02x')
                text_list.append(hex_byte)
            if len(text_list) < packet_length:
                text_list.extend(['00'] * (packet_length - len(text_list)))
            text_data.append(text_list)

    data = np.array(data)
    text_data = np.array(text_data)
    predicts = np.array(predicts)
    true_labels = np.array(true_labels)
    print(data.shape)
    print(text_data.shape)
    print(predicts.shape)
    print(true_labels.shape)

    data = np.insert(data, 0, 0, axis=1)  # 在第0列插入0，后续填充白色
    text_data = np.insert(text_data, 0, predicts, axis=1)

    # 如何再插入一列，填充真实标签？ answer：在第0列插入真实标签
    data = np.insert(data, 0, 0, axis=1)
    text_data = np.insert(text_data, 0, true_labels, axis=1)


    # 截取 data 与 text_data 中 start ~ end 部分数据
    data = data[start:end]
    text_data = text_data[start:end]

    # 自动调整 figsize
    fig_width = max(8, packet_length // 3)  # 每3个字节一个单位宽度，最小宽度为8
    fig_height = max(2, (end - start) // 5)  # 每5个字节一个单位高度，最小高度为2
    plt.figure(figsize=(fig_width, fig_height))

    plt.imshow(data, cmap='Reds', aspect='auto')  # 将 cmap 改为 'Reds' 红色调
    plt.colorbar()
    plt.xlabel('Packets')
    plt.ylabel('Bytes')
    plt.title('Heatmap with Data')

    # 在每个方格中填充数据，并根据注意力权重设置字体颜色
    for i in range(text_data.shape[0]):
        for j in range(text_data.shape[1]):
            weight = data[i, j]  # 获取当前字节的注意力权重
            if weight > attention_threshold:  # 如果注意力权重大于阈值，字体显示为白色
                plt.text(j, i, text_data[i, j], ha='center', va='center', color='black')
            else:
                plt.text(j, i, text_data[i, j], ha='center', va='center', color='black')

    plt.savefig(output_path)
    plt.show()


@click.command()
@click.option(
    "-m",
    "--model_package",
    help='model. Option: "avocado" or "custom..."',
    default="avocado",
)
@click.option("-f", "--file", type=str, help="Path to the data file",
              default="datasets/CLA-M221/pcap/dataExecution+noisePadding/testPLC_attack+unknown.pcapng")
@click.option("-t", "--file_type", type=str, help="type of the files to be processed [pcap, parquet]",
              default="pcap")
@click.option("-w", "--weights_path", type=str, help="Path to the weights file",
              default="outputs/avocado/model.pth")
@click.option("-o", "--output_dir", type=str, help="Path to the output dir",
              default="outputs/avocado/")
@click.option("-d", "--dimension", type=int, help="Dimension of the heatmap", default=0)
@click.option("-n", "--num_class", type=int, help="Number of classes", default=3)
def main(model_package, file, file_type, weights_path, output_dir, dimension, num_class):
    if file_type == "pcap":
        data_list = transform_pcap(file)
    elif file_type == "parquet":
        data_list = transform_parquet(file)

    model = load_model(model_package, weights_path)
    results, accuracy = infer_model(model, data_list, num_class=num_class, dimension=dimension)
    print(f"Model Accuracy: {accuracy}")
    generate_heatmap(results, os.path.join(output_dir, "heatmap.svg"), 1000, 1240)


if __name__ == "__main__":
    main()
