#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将自定义格式的数据转换为GraphGPT评估指令格式
"""

import json
import argparse
from typing import List, Dict, Any


# Cora数据集的7个类别
CORA_CATEGORIES = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
]


def extract_nodes_from_edges(edge_index: List[List[int]]) -> List[int]:
    """
    从边索引中提取所有唯一节点

    Args:
        edge_index: 边索引列表 [[source_nodes], [target_nodes]]

    Returns:
        排序后的唯一节点列表
    """
    all_nodes = set()
    for edge_list in edge_index:
        all_nodes.update(edge_list)
    return sorted(list(all_nodes))


def create_category_question() -> str:
    """
    创建分类问题的提示文本

    Returns:
        包含7个Cora类别的问题字符串
    """
    categories_text = ", ".join([f"{i+1}. {cat}" for i, cat in enumerate(CORA_CATEGORIES)])

    question = f"""Given a citation graph:
<graph>
where the 0th node is the target paper, with the following information:
Abstract: {{abstract}}
Question: Which of the following categories does this paper belong to: {categories_text}? Give the most likely category and explain your reasoning step by step."""

    return question


def convert_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换单个数据项

    Args:
        item: 输入数据项，包含id, graph, summary字段

    Returns:
        转换后的GraphGPT格式数据项
    """
    node_id = item["id"]
    edge_index = item["graph"]
    summary = item["summary"]

    # 提取节点列表
    node_list = extract_nodes_from_edges(edge_index)

    # 找到目标节点在node_list中的索引（假设id就是目标节点）
    node_idx = node_id

    # 创建问题文本
    question_template = create_category_question()
    question = question_template.replace("{abstract}", summary)

    # 构建转换后的数据结构
    converted_item = {
        "id": f"cora_test_{node_id}",
        "graph": {
            "node_idx": node_idx,
            "edge_index": edge_index,
            "node_list": node_list
        },
        "conversations": [
            {
                "from": "human",
                "value": question
            },
            {
                "from": "gpt",
                "value": "基于论文的标题和摘要，我们可以按以下方式对其进行分类：\n\n[这里需要根据实际内容进行分类推理]"
            }
        ]
    }

    return converted_item


def convert_dataset(input_file: str, output_file: str):
    """
    转换整个数据集

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    # 读取输入数据
    print(f"正在读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    # 如果输入是单个对象，转换为列表
    if isinstance(input_data, dict):
        input_data = [input_data]

    print(f"找到 {len(input_data)} 个数据项")

    # 转换每个数据项
    converted_data = []
    for idx, item in enumerate(input_data):
        try:
            converted_item = convert_single_item(item)
            converted_data.append(converted_item)
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(input_data)} 个数据项")
        except Exception as e:
            print(f"处理第 {idx} 个数据项时出错: {e}")
            continue

    # 写入输出文件
    print(f"正在写入输出文件: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成! 成功转换 {len(converted_data)} 个数据项")
    print(f"\nCora数据集类别:")
    for i, cat in enumerate(CORA_CATEGORIES, 1):
        print(f"  {i}. {cat}")


def main():
    parser = argparse.ArgumentParser(
        description='将自定义格式转换为GraphGPT评估指令格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python convert_to_graphgpt_format.py -i input.json -o output.json

输入格式:
  [
    {
      "id": 0,
      "graph": [[source_nodes], [target_nodes]],
      "summary": "论文摘要内容..."
    }
  ]

输出格式:
  GraphGPT评估指令格式，包含7个Cora类别
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='输入JSON文件路径'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='输出JSON文件路径'
    )

    args = parser.parse_args()

    # 执行转换
    convert_dataset(args.input, args.output)


if __name__ == "__main__":
    main()