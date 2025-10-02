import importlib
import os
import click
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# 读取 JSON 文件中的默认参数
def load_defaults(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


# 动态创建选项
def add_options(options):
    def decorator(f):
        for option in options:
            f = click.option(
                f"--{option['name']}",
                default=option.get('default', None),
                help=option.get('help', '')
            )(f)
        return f

    return decorator


@click.command()
@click.option(
    "-m",
    "--model_name",
    help='model. Option: "avocado" or "custom..."',
    default="avocado",
)
@click.option(
    "-t",
    "--task_name",
    help='classification task. Option: "app" or "traffic"',
    required=True,
)
@click.option(
    "-d",
    "--data_dir",
    help="training data dir path containing parquet files",
    required=True,
)
@click.option(
    "-l",
    "--log_dir",
    help="log directory",
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="output dir path to save the model",
    required=True,
)
@click.option(
    "-c",
    "--config",
    help="config file",
    required=True,
)
@click.option(
    "-e",
    "--epoch",
    help="number of epochs",
)
@click.option(
    "-dp",
    "--dropout",
    help="dropout rate",
)
@click.option(
    "-b",
    "--batch_size",
    help="batch size",
)
@click.option(
    "-lr",
    "--learning_rate",
    help="learning rate",
)
@click.option(
    "-l2",
    "--l2_regularization",
    help="L2 regularization",
)
def main(**kwargs):
    # 从命令行参数中获取配置文件路径
    global task_name, model_name
    config_file = kwargs.get('config')

    # 加载 JSON 配置文件中的默认值
    config_defaults = load_defaults(config_file) if config_file else {}

    # 合并 JSON 配置文件中的默认值和命令行参数
    final_config = {**config_defaults, **{k: v for k, v in kwargs.items() if v is not None}}

    try:
        # 加载指定模型的模块
        model_name = final_config.get("model_name")
        task_name = final_config.get("task_name")

        print(f"Loading model module: ml.{model_name}.task")
        model_module = importlib.import_module(f'ml.{model_name}.task')

        # 获取任务函数
        task_function = getattr(model_module, task_name)

    except ModuleNotFoundError:
        raise ValueError(f"Model '{model_name}' not found.")
    except AttributeError:
        raise ValueError(f"Task '{task_name}' not found in model '{model_name}'.")

    required_params = {
        'data_dir': final_config.get('data_dir'),
        'output_dir': final_config.get('output_dir'),
        'log_dir': final_config.get('log_dir'),
        'epoch': final_config.get('epoch'),
        'dropout_rate': final_config.get('dropout_rate'),
        'batch_size': final_config.get('batch_size'),
        'l2_regularization': final_config.get('l2_regularization'),
        'learning_rate': final_config.get('learning_rate'),
        'packet_count': final_config.get('packet_count'),
        'packet_length': final_config.get('packet_length'),
        'label_count': final_config.get('label_count'),
        'attention_dim': final_config.get('attention_dim'),
        'c1_output_dim': final_config.get('c1_output_dim'),
        'c1_kernel_size': final_config.get('c1_kernel_size'),
        'c1_stride': final_config.get('c1_stride'),
        'c2_output_dim': final_config.get('c2_output_dim'),
        'c2_kernel_size': final_config.get('c2_kernel_size'),
        'c2_stride': final_config.get('c2_stride'),
        'fc1_hidden_size': final_config.get('fc1_hidden_size'),
        'fc2_hidden_size': final_config.get('fc2_hidden_size'),
        'fc3_hidden_size': final_config.get('fc3_hidden_size')
    }
    print(required_params)

    task_function(**required_params)


if __name__ == "__main__":
    main()
