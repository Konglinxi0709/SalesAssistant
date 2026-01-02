from typing import Any, Optional
import pandas as pd
import json
import itertools
import numpy as np

def run_consumer_preprocess(input_file, output_file, max_total_product: Optional[int] = None):
    """
    参数:
    - input_file: 输入的CSV文件路径（with_proportion_data.csv）
    - output_file: 输出的CSV文件路径（consumer_data.csv）
    - max_total_product: 可选，最多处理前max_total_product个产品（基于唯一的product_uniq_id）
    """
    # 读取输入文件
    df = pd.read_csv(input_file)

    # 把原始uniq_id字段重命名为product_uniq_id
    if 'uniq_id' in df.columns:
        df = df.rename(columns={'uniq_id': 'product_uniq_id'})

    # 如果有产品个数的限制，筛选前max_total_product个唯一产品
    if max_total_product is not None:
        # 首先确保按原顺序保留出现的product_uniq_id
        product_ids = df['product_uniq_id'].drop_duplicates().head(max_total_product).tolist()
        df = df[df['product_uniq_id'].isin(product_ids)]

    # 存储所有消费者数据
    consumer_rows = []

    # 遍历每一行
    for idx, row in df.iterrows():
        product_uniq_id = row.get('product_uniq_id', '')

        # 解析segmentation_result
        segmentation_result_str = row.get('segmentation_result', '')
        if not segmentation_result_str:
            continue

        try:
            segmentation_data = json.loads(segmentation_result_str)
        except json.JSONDecodeError:
            print(f"警告: 行 {idx} (product_uniq_id: {product_uniq_id}) 的segmentation_result JSON解析失败，跳过")
            continue

        user_group = segmentation_data.get('user_group', '')
        segmentations = segmentation_data.get('segmentations', [])

        if not segmentations:
            continue

        # 解析proportion_estimate
        proportion_estimate_str = row.get('proportion_estimate', '')
        if not proportion_estimate_str:
            continue

        try:
            proportion_data = json.loads(proportion_estimate_str)
        except json.JSONDecodeError:
            print(f"警告: 行 {idx} (product_uniq_id: {product_uniq_id}) 的proportion_estimate JSON解析失败，跳过")
            continue

        # 为每个维度找到非0占比的部分及其索引
        # 每个维度有三个部分：0=负价值, 1=无价值, 2=正价值
        dimension_options = []  # 每个元素包含维度信息和可选部分列表

        for dim_idx, seg in enumerate(segmentations):
            dim_name = seg.get('dimension_name', '')

            # 找到对应的占比数据
            prop_item = None
            for p in proportion_data:
                if p.get('dimension_name', '') == dim_name:
                    prop_item = p
                    break

            if prop_item is None:
                print(f"警告: 行 {idx} (product_uniq_id: {product_uniq_id}) 的维度 {dim_name} 在proportion_estimate中未找到，跳过该维度")
                continue

            proportions = prop_item.get('proportions', [])
            if len(proportions) != 3:
                print(f"警告: 行 {idx} (product_uniq_id: {product_uniq_id}) 的维度 {dim_name} 的proportions格式不正确，跳过")
                continue

            # 找到非0占比的部分（部分索引保持为原始三部分的索引0,1,2）
            part_options = []
            for part_idx in range(3):
                prop_val = proportions[part_idx]
                # 检查是否非0（容忍小的浮点误差）
                if isinstance(prop_val, (int, float)) and abs(float(prop_val)) > 1e-9:
                    # 获取对应的判定标准
                    if part_idx == 0:
                        criteria = seg.get('negative_value_criteria', '')
                    elif part_idx == 1:
                        criteria = seg.get('neutral_value_criteria', '')
                    else:  # part_idx == 2
                        criteria = seg.get('positive_value_criteria', '')

                    # 只有判定依据不为空才添加（空集判定依据在market_segment中设为空字符串）
                    if criteria:  # 忽略空字符串，因为那代表空集
                        part_options.append((part_idx, criteria))

            if part_options:
                dimension_options.append({
                    'dimension_index': dim_idx,
                    'dimension_name': dim_name,
                    'part_options': part_options
                })

        if not dimension_options:
            print(f"警告: 行 {idx} (product_uniq_id: {product_uniq_id}) 没有有效的维度选项，跳过")
            continue

        # 生成所有可能的组合（每个维度选一个部分）
        # 构建所有可能的组合
        part_option_lists = [dim['part_options'] for dim in dimension_options]
        all_combinations = list(itertools.product(*part_option_lists))

        # 为每个组合创建一个消费者行
        for combo_idx, combination in enumerate(all_combinations):
            # 构建uniq_id: product_uniq_id_维度1部分索引_维度2部分索引_...
            part_indices = [str(part_idx) for part_idx, _ in combination]
            uniq_id = f"{product_uniq_id}_{'_'.join(part_indices)}"

            # 构建消费者群体定义文本为XML风格
            definition_parts = [f"<potential_user_group>\n{user_group}\n</potential_user_group>"]

            for dim_idx, (part_idx, criteria) in enumerate(combination):
                dim_name = dimension_options[dim_idx]['dimension_name']
                # 根据part_idx确定部分类型：0=负价值, 1=无价值, 2=正价值
                part_type_map = {0: "负价值", 1: "无价值", 2: "正价值"}
                part_type = part_type_map[part_idx]
                # 对criteria内容做转义可以进一步处理（如需严格XML），此处直接格式化
                definition_parts.append(
                    f"<Dimension index=\"{dim_idx + 1}\" name=\"{dim_name}\">\n"
                    f"【属于该部分的条件】{criteria}\n"
                    f"</Dimension>"
                )

            consumer_definition = "<consumer_group_definition>\n" + "\n".join(definition_parts) + "\n</consumer_group_definition>"


            # 创建新行，复制原行的所有数据
            new_row = row.copy()
            new_row['uniq_id'] = uniq_id  # 替换成新的uniq_id
            new_row['consumer_definition'] = consumer_definition
            new_row['dimensions_count'] = len(dimension_options)  # 用于排序

            consumer_rows.append(new_row)

    if not consumer_rows:
        print("警告: 没有生成任何消费者数据")
        return

    # 创建DataFrame
    consumer_df = pd.DataFrame(consumer_rows)

    # 排序：
    # 1. 主标准：按所属产品的维度个数从少到多
    # 2. 副标准：按产品内各个消费者群体的编号顺序从小到大（即按uniq_id排序，因为uniq_id包含了维度索引）
    consumer_df = consumer_df.sort_values(
        by=['dimensions_count', 'uniq_id'],
        ascending=[True, True]
    )

    # 删除临时排序用的dimensions_count列（如果不需要保留的话）
    # 如果需要保留该列可以注释掉下一行
    # consumer_df = consumer_df.drop(columns=['dimensions_count'], errors='ignore')

    # 保存到输出文件
    consumer_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"成功生成 {len(consumer_df)} 行消费者数据，保存到 {output_file}")

