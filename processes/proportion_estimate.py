import json
import pandas as pd
from typing import Dict, Any, Callable
from utils.call_cuc_llm import call_reasoner

async def proportion_estimate_task(row: Dict[str, Any], logger: Callable[[str], None]) -> Dict[str, Any]:
    """人群占比估计任务回调函数。"""
    uniq_id = row.get('uniq_id', 'unknown_id')
    product_name = row.get('modified_name', '')
    
    logger(f"[{uniq_id}] 开始人群占比估计: {product_name}")
    
    # 从segmentation_result中提取信息
    segmentation_result_str = row.get('segmentation_result', '')
    if not segmentation_result_str:
        raise ValueError("segmentation_result 字段为空")
    
    try:
        segmentation_data = json.loads(segmentation_result_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"segmentation_result JSON解析失败: {str(e)}")
    
    user_group = segmentation_data.get('user_group', '')
    segmentations = segmentation_data.get('segmentations', [])
    
    if not user_group:
        raise ValueError("segmentation_result 中缺少 user_group 字段")
    if not segmentations or len(segmentations) == 0:
        raise ValueError("segmentation_result 中 segmentations 为空")
    
    logger(f"[{uniq_id}] 提取到 {len(segmentations)} 个维度")
    
    # 组装信息用于提示词
    dimensions_info = ""
    for idx, seg in enumerate(segmentations):
        dim_name = seg.get('dimension_name', '')
        # 兼容新旧格式：优先使用新的同向/反向属性格式，如果没有则使用旧的attribute_names
        positive_attr_names = seg.get('positive_attribute_names', [])
        negative_attr_names = seg.get('negative_attribute_names', [])
        if not positive_attr_names:
            # 兼容旧格式
            attr_names = seg.get('attribute_names', [])
            positive_attr_names = attr_names
            negative_attr_names = []
        
        # 合并所有属性名称用于显示
        all_attr_names = positive_attr_names + negative_attr_names
        attr_display = []
        if positive_attr_names:
            attr_display.append(f"同向属性: {', '.join(positive_attr_names)}")
        if negative_attr_names:
            attr_display.append(f"反向属性: {', '.join(negative_attr_names)}")
        
        negative = seg.get('negative_value_criteria', '')
        neutral = seg.get('neutral_value_criteria', '')
        positive = seg.get('positive_value_criteria', '')
        
        
        # 如果判定依据为空，用特殊标记说明是空集
        negative_label = "[空集]" if negative == '' else negative
        neutral_label = "[空集]" if neutral == '' else neutral
        positive_label = "[空集]" if positive == '' else positive
        
        dimensions_info += f"""
<dimension_{idx + 1}>
<dimension_name>{dim_name}</dimension_name>
<corresponding_attributes>{'; '.join(attr_display) if attr_display else ', '.join(all_attr_names)}</corresponding_attributes>
<negative_value_criteria>{negative_label}</negative_value_criteria>
<neutral_value_criteria>{neutral_label}</neutral_value_criteria>
<positive_value_criteria>{positive_label}</positive_value_criteria>
</dimension_{idx + 1}>
"""
    
    # 系统提示词
    system_prompt = """你是一个市场调研专家，专门负责基于用户特征判断估算人群占比。你需要深入思考和分析各个用户维度的特征分布，给出准确的人群占比估计。"""
    
    # 用户提示词
    user_prompt = f"""
<task>
基于提供的潜在用户群体和各用户维度的三个划分标准，估计各个维度上负价值、无价值、正价值三个部分在潜在用户群体整体中的占比。
</task>

<principles>
<proportion_estimation_principles>
1. 你需要深入思考和分析各个维度特征的普遍性。考虑现实社会中各类人群的真实特征分布，不要基于刻板印象或偏见。
2. 仔细分析各维度划分标准中描述的用户特征，思考在现实中符合这些标准的人群规模和相对比例。
3. 考虑不同特征之间的关联性。某些特征可能在人群中更容易同时出现，某些特征可能在人群中相互排斥。
4. 参考统计学知识和生活经验，考虑特征的分布往往存在长尾分布、正态分布、幂律分布等不同模式。
5. 对于每个维度，深入思考负价值、无价值、正价值三个群体在人群中的真实占比关系，考虑特征的稀缺性和普遍性。
6. 占比数字应该基于逻辑推理和合理假设，而非随意猜测。
</proportion_estimation_principles>

<constraints>
1. 每个维度的三个占比之和必须等于1.000
2. 每个占比数字应该为纯小数，保留三位小数
3. 每个占比代表当前维度的当前部分占潜在用户群体整体的多少
4. 占比应尽可能贴近真实世界的特征分布
</constraints>
</principles>

<user_group>
{user_group}
</user_group>

<dimensions>
{dimensions_info}
</dimensions>

<analysis_guidance>
对于每个维度，请按照以下步骤进行分析：
1. 仔细阅读该维度的三个划分标准（negative_value_criteria, neutral_value_criteria, positive_value_criteria）
2. 如果某个判定依据显示为"[空集]"，则代表该部分在潜在用户群体中不存在，你应将该部分的占比估计为0.000
3. 思考在潜在用户群体中，符合每个标准的用户特征的真实分布情况
4. 考虑该维度特征在人群中的普遍程度（是大多数人的特征还是少数人的特征）
5. 分析三个部分可能的人群占比关系
6. 结合生活经验和常识，给出合理的占比估计

注意：
- 不要简单地假设三个部分各占1/3
- 考虑实际情况中，某些特征可能更普遍，某些特征可能更罕见
- 负价值、无价值、正价值三个群体的大小关系需要结合实际的特征分布进行推理
- 如果某个判定依据为空集（标记为"[空集]"），该部分占比必须为0.000
</analysis_guidance>

<output_format>
输出必须是数组格式，直接给出M个对象的数组，每个对象包含以下字段：
- dimension_name: 维度名称
- proportions: 包含三个浮点数的数组，分别代表负价值、无价值、正价值的占比

输出示例：
[
    {{"dimension_name": "维度1名称", "proportions": [0.150, 0.600, 0.250]}},
    {{"dimension_name": "维度2名称", "proportions": [0.200, 0.500, 0.300]}},
    ...
]

注意：
1. proportions数组必须有且仅有三个元素
2. 三个元素分别代表negative_value、neutral_value、positive_value的占比
3. 三个元素之和必须等于1.000
4. 所有数字保留三位小数
5. 直接输出数组，不要使用```json```或``````包裹
6. 所有的输出必须使用中文
</output_format>
"""
    
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    
    # 调用大模型
    reasoning_text, output_text = await call_reasoner(system_prompt, user_prompt, reasoning_effort="low", debug=False)
    logger(f"[{uniq_id}] LLM思考过程: \n{reasoning_text}")
    logger(f"[{uniq_id}] LLM输出原文: \n{output_text}")
    
    # 解析JSON输出
    try:
        output_json = json.loads(output_text)
        
        # 验证输出结构
        if not isinstance(output_json, list):
            raise ValueError("输出必须是一个数组")
        
        if len(output_json) != len(segmentations):
            raise ValueError(f"输出数组长度 {len(output_json)} 与维度数量 {len(segmentations)} 不匹配")
        
        dimension_names_from_seg = [seg.get('dimension_name', '') for seg in segmentations]
        dimension_names_from_output = [item.get('dimension_name', '') for item in output_json]
        
        # 验证每个对象的结构
        for i, (seg, item) in enumerate(zip(segmentations, output_json)):
            if not isinstance(item, dict):
                raise ValueError(f"输出[{i}] 不是字典类型")
            
            if "dimension_name" not in item:
                raise ValueError(f"输出[{i}] 缺少字段: dimension_name")
            if "proportions" not in item:
                raise ValueError(f"输出[{i}] 缺少字段: proportions")
            
            proportions = item["proportions"]
            if not isinstance(proportions, list):
                raise ValueError(f"输出[{i}].proportions 不是数组类型")
            
            if len(proportions) != 3:
                raise ValueError(f"输出[{i}].proportions 必须有且仅有3个元素，当前为{len(proportions)}")
            
            # 获取当前维度的判定依据
            negative_criteria = seg.get('negative_value_criteria', '')
            neutral_criteria = seg.get('neutral_value_criteria', '')
            positive_criteria = seg.get('positive_value_criteria', '')
            
            for j, prop in enumerate(proportions):
                if not isinstance(prop, (int, float)):
                    raise ValueError(f"输出[{i}].proportions[{j}] 不是数字类型")
            
            # 验证空集对应的占比必须为0
            if negative_criteria == '' and abs(proportions[0]) > 0.0001:
                raise ValueError(f"输出[{i}].proportions[0] (negative_value) 应改为 0.000，因为判定依据为空集")
            if neutral_criteria == '' and abs(proportions[1]) > 0.0001:
                raise ValueError(f"输出[{i}].proportions[1] (neutral_value) 应改为 0.000，因为判定依据为空集")
            if positive_criteria == '' and abs(proportions[2]) > 0.0001:
                raise ValueError(f"输出[{i}].proportions[2] (positive_value) 应改为 0.000，因为判定依据为空集")
            
            # 验证三个占比之和是否等于1
            prop_sum = round(sum(proportions), 3)
            if abs(prop_sum - 1.0) > 0.0001:
                raise ValueError(f"输出[{i}].proportions 三个占比之和为 {prop_sum}，必须等于 1.000")
        
        # 验证dimension_name是否匹配
        for i, (seg, output) in enumerate(zip(segmentations, output_json)):
            expected_name = seg.get('dimension_name', '')
            actual_name = output.get('dimension_name', '')
            if expected_name != actual_name:
                raise ValueError(f"维度名称不匹配: 期望 '{expected_name}'，实际 '{actual_name}'")
        
        logger(f"[{uniq_id}] 人群占比估计成功")
        
    except json.JSONDecodeError as e:
        logger(f"[{uniq_id}] JSON解析失败: {str(e)}")
        raise ValueError(f"[人群占比估计]JSON解析失败: {str(e)}")
    except Exception as e:
        logger(f"[{uniq_id}] 人群占比估计失败: {str(e)}")
        raise ValueError(f"[人群占比估计]失败: {str(e)}")
    
    # 更新行数据，添加新字段
    output_row = row.copy()
    output_row['proportion_estimate'] = json.dumps(output_json, ensure_ascii=False)
    
    logger(f"[{uniq_id}] **人群占比估计完成**")
    logger(f"[{uniq_id}] 占比估计结果: {json.dumps(output_json, ensure_ascii=False, indent=2)}")
    
    return output_row

def filter_existing_proportion(row: Dict[str, Any], existing_output_df: pd.DataFrame) -> bool:
    """判断某一行是否需要人群占比估计。如果uniq_id已存在于输出df中且已有proportion_estimate字段，则不需要估计。"""
    uniq_id = row.get('uniq_id')
    if uniq_id is None:
        # 如果没有uniq_id，默认处理它
        return True 
    
    # 如果输出DataFrame为空，则所有行都需要处理
    if existing_output_df.empty:
        return True

    # 检查uniq_id是否在现有输出文件中，且是否已有proportion_estimate字段
    if uniq_id in existing_output_df['uniq_id'].values:
        # 找到对应的行
        matching_rows = existing_output_df[existing_output_df['uniq_id'] == uniq_id]
        if not matching_rows.empty:
            # 检查是否已有proportion_estimate字段且不为空
            if 'proportion_estimate' in matching_rows.columns:
                proportion_value = matching_rows.iloc[0]['proportion_estimate']
                if pd.notna(proportion_value) and str(proportion_value).strip() != '':
                    return False  # 已有占比估计，不需要重新估计
    
    return True  # 需要估计
