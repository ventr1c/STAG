import torch
import os

def merge_pytorch_datasets(arxiv_path, pubmed_path, output_path):
    """
    加载两个.pt数据集文件，将它们合并到一个字典中，并保存为新的.pt文件。

    Args:
        arxiv_path (str): 'ogbn-arxiv.pt' 文件的路径。
        pubmed_path (str): 'pubmed.pt' 文件的路径。
        output_path (str): 合并后输出文件的路径。
    """
    # 检查输入文件是否存在
    if not os.path.exists(arxiv_path):
        print(f"错误: 文件未找到 '{arxiv_path}'")
        return
    if not os.path.exists(pubmed_path):
        print(f"错误: 文件未找到 '{pubmed_path}'")
        return

    print(f"正在加载数据集: '{arxiv_path}'...")
    try:
        arxiv_data = torch.load(arxiv_path)
        print("ogbn-arxiv 数据集加载成功。")
    except Exception as e:
        print(f"加载 '{arxiv_path}' 时出错: {e}")
        return

    print(f"正在加载数据集: '{pubmed_path}'...")
    try:
        pubmed_data = torch.load(pubmed_path)
        print("pubmed 数据集加载成功。")
    except Exception as e:
        print(f"加载 '{pubmed_path}' 时出错: {e}")
        return

    # 创建一个包含两个数据集的字典
    combined_data = {
        'arxiv': arxiv_data,
        'pubmed': pubmed_data
    }
    print("\n已创建合并字典。")

    # 将合并后的字典保存到新文件
    print(f"正在将合并后的数据保存到: '{output_path}'...")
    try:
        torch.save(combined_data, output_path)
        print(f"成功！合并后的文件已保存为 '{output_path}'。")
    except Exception as e:
        print(f"保存到 '{output_path}' 时出错: {e}")

if __name__ == '__main__':
    # 定义输入和输出文件名
    arxiv_file = '/root/autodl-tmp/GraphCLIP/processed_data/ogbn-arxiv.pt'
    pubmed_file = '/root/autodl-tmp/GraphCLIP/processed_data/pubmed.pt'
    output_file = '/root/autodl-tmp/GraphCLIP/processed_data/graph_data.pt'

    # 执行合并函数
    merge_pytorch_datasets(arxiv_file, pubmed_file, output_file)