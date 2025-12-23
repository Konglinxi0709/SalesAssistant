import json
import re
import os
import pandas as pd
import asyncio
from typing import Any, Callable, Dict, List
from utils.call_cuc_llm import call_reasoner


def _parse_group_dim_values_from_uniq_id(uniq_id: str, num_dims: int) -> List[int] | None:
    """从uniq_id末尾解析维度取值(0/1/2)。"""
    parts = str(uniq_id).split("_")
    if len(parts) < num_dims:
        return None
    try:
        return [int(x) for x in parts[-num_dims:]]
    except Exception:
        return None


def _group_value_label(v: int) -> str:
    """将群体维度取值(0/1/2)映射为中文标签。"""
    if v == 0:
        return "负价值"
    if v == 1:
        return "无价值"
    if v == 2:
        return "正价值"
    return str(v)


def _calc_group_proportion(proportion_estimate: list, dim_values_012: list[int]) -> float:
    """
    计算群体占比 N(x)=∏ φ_i(x_i)，其中dim_values_012为0/1/2（对应负/无/正）。
    proportion_estimate结构来自consumer_analysis.csv中的proportion_estimate字段。
    """
    if not isinstance(proportion_estimate, list):
        return 0.0
    prop = 1.0
    for i, v in enumerate(dim_values_012):
        try:
            probs = proportion_estimate[i]["proportions"]
            prop *= float(probs[v])
        except Exception:
            return 0.0
    return float(prop)


def safe_json_loads(x):
    """安全地解析JSON字符串"""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    return x


def parse_price(price_str):
    """
    解析价格字符串，移除货币符号并转换为浮点数
    例如: "485.60￥" -> 485.60
    """
    if isinstance(price_str, (int, float)):
        val = float(price_str)
        return val if val > 0 else 1.0
    if not isinstance(price_str, str):
        return 1.0
    try:
        import re
        clean_str = re.sub(r'[^\d.]', '', price_str)
        val = float(clean_str)
        return val if val > 0 else 1.0
    except:
        return 1.0


def generate_optimization_table(product_row: pd.Series) -> List[Dict[str, Any]]:
    """
    为单个产品生成属性优化优先顺序表
    
    返回优化选项列表，按重要性降序排序
    """
    # 解析数据
    prop_est = safe_json_loads(product_row.get('proportion_estimate'))
    beta_plus_list = safe_json_loads(product_row.get('beta_plus_coefficients')) or []
    beta_minus_list = safe_json_loads(product_row.get('beta_minus_coefficients')) or []
    segmentation = safe_json_loads(product_row.get('segmentation_result'))
    users_data = safe_json_loads(product_row.get('users_data', '[]'))
    optimal_price = float(product_row.get('optimal_price', 0.0))
    
    if not isinstance(users_data, list):
        users_data = []
    
    dims = segmentation.get('segmentations', []) if segmentation else []
    
    # 构建属性到维度的映射
    attr_map = {}
    attr_desc_map = {}
    
    if segmentation:
        # 提取属性描述
        for item in segmentation.get('total_attributes', []):
            attr_desc_map[item['attribute_name']] = item['attribute_description']
        
        # 构建属性映射
        for idx, dim in enumerate(dims):
            dim_name = dim.get('dimension_name', f"Dimension {idx+1}")
            beta_plus_val = beta_plus_list[idx] if idx < len(beta_plus_list) else 0.0
            beta_minus_val = beta_minus_list[idx] if idx < len(beta_minus_list) else 0.0
            
            for attr in dim.get('positive_attribute_names', []):
                attr_map[attr] = {
                    'dim_index': idx,
                    'dim_name': dim_name,
                    'beta_plus': beta_plus_val,
                    'beta_minus': beta_minus_val,
                    'direction': '正向'
                }
            for attr in dim.get('negative_attribute_names', []):
                attr_map[attr] = {
                    'dim_index': idx,
                    'dim_name': dim_name,
                    'beta_plus': beta_plus_val,
                    'beta_minus': beta_minus_val,
                    'direction': '反向'
                }
    
    # 检查维度取值覆盖情况
    num_dims = len(beta_plus_list)
    has_pos = [False] * num_dims
    has_neg = [False] * num_dims
    
    for user_data in users_data:
        gid = str(user_data.get('user_uniq_id', ''))
        dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
        if dim_vals:
            for i, v in enumerate(dim_vals):
                if v == 2 and i < len(has_pos):
                    has_pos[i] = True
                if v == 0 and i < len(has_neg):
                    has_neg[i] = True
    
    # 计算每个维度各取值的占比
    dim_proportions = {}
    if isinstance(prop_est, list):
        for dim_idx in range(len(prop_est)):
            if dim_idx < len(prop_est):
                dim_prop_info = prop_est[dim_idx]
                if isinstance(dim_prop_info, dict) and 'proportions' in dim_prop_info:
                    props = dim_prop_info['proportions']
                    if isinstance(props, list) and len(props) >= 3:
                        dim_proportions[dim_idx] = {
                            0: float(props[0]) if props[0] is not None else 0.0,
                            1: float(props[1]) if props[1] is not None else 0.0,
                            2: float(props[2]) if props[2] is not None else 0.0
                        }
    
    # 收集所有属性的优化选项
    optimization_options = []
    
    for attr_name, attr_info in attr_map.items():
        dim_idx = attr_info.get('dim_index')
        if dim_idx is None or dim_idx >= len(beta_plus_list):
            continue
        
        beta_plus_val = beta_plus_list[dim_idx] if dim_idx < len(beta_plus_list) else 0.0
        beta_minus_val = beta_minus_list[dim_idx] if dim_idx < len(beta_minus_list) else 0.0
        dim_prop = dim_proportions.get(dim_idx, {0: 0.0, 1: 0.0, 2: 0.0})
        
        # 获取该属性的当前表现
        attr_effect = attr_desc_map.get(attr_name, '（无描述）')
        attr_direction = attr_info.get('direction', '正向')
        
        # 获取维度判定标准
        dim_info = dims[dim_idx] if dim_idx < len(dims) else None
        neg_criteria = dim_info.get('negative_value_criteria', '') if dim_info else ''
        neu_criteria = dim_info.get('neutral_value_criteria', '') if dim_info else ''
        pos_criteria = dim_info.get('positive_value_criteria', '') if dim_info else ''
        
        # 选项1：对负价值群体减少负面影响
        if dim_prop[0] > 0 and has_neg[dim_idx] if dim_idx < len(has_neg) else False:
            importance = dim_prop[0] * abs(beta_minus_val)
            
            if attr_direction == '正向':
                opt_direction = '对负价值群体减少负面影响'
                opt_criteria = neg_criteria
            else:
                opt_direction = '对正价值群体提高满意度'
                opt_criteria = pos_criteria
            
            # 查找参考优化建议
            matching_groups = []
            for user_row in users_data:
                gid = str(user_row.get('user_uniq_id', ''))
                group_dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
                if group_dim_vals is not None and dim_idx < len(group_dim_vals):
                    if group_dim_vals[dim_idx] == 0:  # 负价值
                        price_str = user_row.get('psychological_price', '0')
                        valuation = parse_price(price_str)
                        
                        aa = user_row.get('attribute_analysis')
                        if isinstance(aa, str):
                            aa = safe_json_loads(aa)
                        elif not isinstance(aa, list):
                            aa = safe_json_loads(str(aa)) if aa else []
                        
                        attr_score = None
                        if isinstance(aa, list):
                            for item in aa:
                                if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                    attr_score = item.get('attribute_score')
                                    break
                        
                        if attr_score is not None:
                            matching_groups.append({
                                'gid': gid,
                                'valuation': valuation,
                                'attr_score': attr_score,
                                'user_row': user_row
                            })
            
            # 排序并取第一个
            reference_suggestion = "（无参考建议）"
            reference_gid = "（无）"
            reference_profile = "（无）"
            reference_score = "（无）"
            reference_valuation = "（无）"
            
            if matching_groups:
                matching_groups.sort(key=lambda g: (
                    abs(g['valuation'] - optimal_price),
                    g['attr_score']
                ))
                top_group = matching_groups[0]
                reference_gid = top_group['gid']
                reference_score = str(top_group['attr_score'])
                reference_valuation = f"{top_group['valuation']:.2f}"
                reference_profile = str(top_group['user_row'].get('user_profile', '（无）'))
                
                top_aa = top_group['user_row'].get('attribute_analysis')
                if isinstance(top_aa, str):
                    top_aa = safe_json_loads(top_aa)
                elif not isinstance(top_aa, list):
                    top_aa = safe_json_loads(str(top_aa)) if top_aa else []
                
                if isinstance(top_aa, list):
                    for item in top_aa:
                        if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                            ref_sug = item.get('attribute_optimization_suggestion', '')
                            if ref_sug:
                                reference_suggestion = ref_sug
                            break
            
            optimization_options.append({
                'priority': 0,  # 稍后排序
                'attr_name': attr_name,
                'direction': opt_direction,
                'current_performance': attr_effect,
                'user_criteria': opt_criteria,
                'importance_score': importance,
                'reference_suggestion': reference_suggestion,
                'reference_gid': reference_gid,
                'reference_user_profile': reference_profile,
                'reference_score': reference_score,
                'reference_valuation': reference_valuation
            })
        
        # 选项2：对无价值群体增强吸引力
        if dim_prop[1] > 0:
            has_all = (has_neg[dim_idx] if dim_idx < len(has_neg) else False) and \
                      (has_pos[dim_idx] if dim_idx < len(has_pos) else False)
            
            if has_all:
                beta_avg = (beta_plus_val + beta_minus_val) / 2.0
                importance = dim_prop[1] * abs(beta_avg)
                beta_used = beta_avg
            else:
                if has_neg[dim_idx] if dim_idx < len(has_neg) else False and abs(beta_minus_val) > 1e-12:
                    importance = dim_prop[1] * abs(beta_minus_val)
                    beta_used = beta_minus_val
                elif has_pos[dim_idx] if dim_idx < len(has_pos) else False and abs(beta_plus_val) > 1e-12:
                    importance = dim_prop[1] * abs(beta_plus_val)
                    beta_used = beta_plus_val
                else:
                    importance = 0.0
                    beta_used = 0.0
            
            if importance > 0:
                # 查找参考优化建议（类似上面的逻辑，但dim_val=1）
                matching_groups = []
                for user_row in users_data:
                    gid = str(user_row.get('user_uniq_id', ''))
                    group_dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
                    if group_dim_vals is not None and dim_idx < len(group_dim_vals):
                        if group_dim_vals[dim_idx] == 1:  # 无价值
                            price_str = user_row.get('psychological_price', '0')
                            valuation = parse_price(price_str)
                            
                            aa = user_row.get('attribute_analysis')
                            if isinstance(aa, str):
                                aa = safe_json_loads(aa)
                            elif not isinstance(aa, list):
                                aa = safe_json_loads(str(aa)) if aa else []
                            
                            attr_score = None
                            if isinstance(aa, list):
                                for item in aa:
                                    if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                        attr_score = item.get('attribute_score')
                                        break
                            
                            if attr_score is not None:
                                matching_groups.append({
                                    'gid': gid,
                                    'valuation': valuation,
                                    'attr_score': attr_score,
                                    'user_row': user_row
                                })
                
                reference_suggestion = "（无参考建议）"
                reference_gid = "（无）"
                reference_profile = "（无）"
                reference_score = "（无）"
                reference_valuation = "（无）"
                
                if matching_groups:
                    matching_groups.sort(key=lambda g: (
                        abs(g['valuation'] - optimal_price),
                        g['attr_score']
                    ))
                    top_group = matching_groups[0]
                    reference_gid = top_group['gid']
                    reference_score = str(top_group['attr_score'])
                    reference_valuation = f"{top_group['valuation']:.2f}"
                    reference_profile = str(top_group['user_row'].get('user_profile', '（无）'))
                    
                    top_aa = top_group['user_row'].get('attribute_analysis')
                    if isinstance(top_aa, str):
                        top_aa = safe_json_loads(top_aa)
                    elif not isinstance(top_aa, list):
                        top_aa = safe_json_loads(str(top_aa)) if top_aa else []
                    
                    if isinstance(top_aa, list):
                        for item in top_aa:
                            if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                ref_sug = item.get('attribute_optimization_suggestion', '')
                                if ref_sug:
                                    reference_suggestion = ref_sug
                                break
                
                optimization_options.append({
                    'priority': 0,
                    'attr_name': attr_name,
                    'direction': '对无价值群体增强吸引力',
                    'current_performance': attr_effect,
                    'user_criteria': neu_criteria,
                    'importance_score': importance,
                    'reference_suggestion': reference_suggestion,
                    'reference_gid': reference_gid,
                    'reference_user_profile': reference_profile,
                    'reference_score': reference_score,
                    'reference_valuation': reference_valuation
                })
        
        # 选项3：对正价值群体提高满意度
        if dim_prop[2] > 0 and has_pos[dim_idx] if dim_idx < len(has_pos) else False:
            importance = dim_prop[2] * abs(beta_plus_val)
            
            if attr_direction == '正向':
                opt_direction = '对正价值群体提高满意度'
                opt_criteria = pos_criteria
            else:
                opt_direction = '对负价值群体减少负面影响'
                opt_criteria = neg_criteria
            
            # 查找参考优化建议（类似上面的逻辑，但dim_val=2）
            matching_groups = []
            for user_row in users_data:
                gid = str(user_row.get('user_uniq_id', ''))
                group_dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
                if group_dim_vals is not None and dim_idx < len(group_dim_vals):
                    if group_dim_vals[dim_idx] == 2:  # 正价值
                        price_str = user_row.get('psychological_price', '0')
                        valuation = parse_price(price_str)
                        
                        aa = user_row.get('attribute_analysis')
                        if isinstance(aa, str):
                            aa = safe_json_loads(aa)
                        elif not isinstance(aa, list):
                            aa = safe_json_loads(str(aa)) if aa else []
                        
                        attr_score = None
                        if isinstance(aa, list):
                            for item in aa:
                                if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                    attr_score = item.get('attribute_score')
                                    break
                        
                        if attr_score is not None:
                            matching_groups.append({
                                'gid': gid,
                                'valuation': valuation,
                                'attr_score': attr_score,
                                'user_row': user_row
                            })
            
            reference_suggestion = "（无参考建议）"
            reference_gid = "（无）"
            reference_profile = "（无）"
            reference_score = "（无）"
            reference_valuation = "（无）"
            
            if matching_groups:
                matching_groups.sort(key=lambda g: (
                    abs(g['valuation'] - optimal_price),
                    g['attr_score']
                ))
                top_group = matching_groups[0]
                reference_gid = top_group['gid']
                reference_score = str(top_group['attr_score'])
                reference_valuation = f"{top_group['valuation']:.2f}"
                reference_profile = str(top_group['user_row'].get('user_profile', '（无）'))
                
                top_aa = top_group['user_row'].get('attribute_analysis')
                if isinstance(top_aa, str):
                    top_aa = safe_json_loads(top_aa)
                elif not isinstance(top_aa, list):
                    top_aa = safe_json_loads(str(top_aa)) if top_aa else []
                
                if isinstance(top_aa, list):
                    for item in top_aa:
                        if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                            ref_sug = item.get('attribute_optimization_suggestion', '')
                            if ref_sug:
                                reference_suggestion = ref_sug
                            break
            
            optimization_options.append({
                'priority': 0,
                'attr_name': attr_name,
                'direction': opt_direction,
                'current_performance': attr_effect,
                'user_criteria': opt_criteria,
                'importance_score': importance,
                'reference_suggestion': reference_suggestion,
                'reference_gid': reference_gid,
                'reference_user_profile': reference_profile,
                'reference_score': reference_score,
                'reference_valuation': reference_valuation
            })
    
    # 按重要性降序排序并设置优先级
    optimization_options.sort(key=lambda x: x['importance_score'], reverse=True)
    for idx, opt in enumerate(optimization_options, 1):
        opt['priority'] = idx
    
    return optimization_options


async def product_optimization_task(row: pd.Series, logger: Callable[[str], None]) -> pd.Series:
    """
    产品优化任务：基于consumer_analysis结果优化产品设计文档
    """
    product_name = row.get('modified_name', '')
    uniq_id = row.get('uniq_id', 'unknown_id')
    
    logger(f"[{uniq_id}] 开始处理产品优化: {product_name}")
    
    # 生成属性优化优先顺序表
    optimization_table = generate_optimization_table(row)
    
    if not optimization_table:
        logger(f"[{uniq_id}] 警告: 无法生成属性优化优先顺序表")
        return None
    
    # 解析产品数据
    segmentation = safe_json_loads(row.get('segmentation_result'))
    prop_est = safe_json_loads(row.get('proportion_estimate'))
    hard_design = str(row.get('hard_design', '') or '')
    core_features = str(row.get('core_features', '') or '')
    value_proposition = str(row.get('value_proposition', '') or '')
    optimal_price = row.get('optimal_price', 0.0)
    
    # 构建提示词
    system_prompt = """你是一名产品经理，负责根据市场反馈的结果对产品进行优化，尽可能提高产品的市场回报率（购买产品的人数*产品的销售价格）。"""
    
    # 构建用户维度信息
    dims_info = ""
    if segmentation and isinstance(prop_est, list):
        dims = segmentation.get('segmentations', [])
        for idx, dim in enumerate(dims):
            dim_name = dim.get('dimension_name', f"维度{idx+1}")
            dim_prop = prop_est[idx] if idx < len(prop_est) else {}
            proportions = dim_prop.get('proportions', [0.0, 0.0, 0.0])
            
            neg_criteria = dim.get('negative_value_criteria', '')
            neu_criteria = dim.get('neutral_value_criteria', '')
            pos_criteria = dim.get('positive_value_criteria', '')
            
            dims_info += f"\n**维度{idx+1}：{dim_name}**\n"
            dims_info += f"- 负价值群体判定标准：{neg_criteria if neg_criteria else '不存在这种用户'}（占比：{proportions[0]:.2%}）\n"
            dims_info += f"- 无价值群体判定标准：{neu_criteria if neu_criteria else '不存在这种用户'}（占比：{proportions[1]:.2%}）\n"
            dims_info += f"- 正价值群体判定标准：{pos_criteria if pos_criteria else '不存在这种用户'}（占比：{proportions[2]:.2%}）\n"
    
    # 构建属性信息
    attrs_info = ""
    if segmentation:
        dims = segmentation.get('segmentations', [])
        total_attrs = segmentation.get('total_attributes', [])
        
        # 构建属性到维度的映射
        attr_dim_map = {}
        for idx, dim in enumerate(dims):
            for attr in dim.get('positive_attribute_names', []):
                attr_dim_map[attr] = (idx, '正向', dim.get('dimension_name', f'维度{idx+1}'))
            for attr in dim.get('negative_attribute_names', []):
                attr_dim_map[attr] = (idx, '反向', dim.get('dimension_name', f'维度{idx+1}'))
        
        for attr in total_attrs:
            attr_name = attr.get('attribute_name', '')
            attr_desc = attr.get('attribute_description', '')
            if attr_name in attr_dim_map:
                dim_idx, direction, dim_name = attr_dim_map[attr_name]
                attrs_info += f"\n- **{attr_name}**：{attr_desc}\n"
                attrs_info += f"  - 对应维度：{dim_name}\n"
                attrs_info += f"  - 相关性：{direction}\n"
    
    # 构建优化优先顺序表（XML格式，完整内容不截断）
    table_xml = "<optimization_priority_table>\n"
    for opt in optimization_table:
        # 转义XML特殊字符
        def escape_xml(text):
            if not isinstance(text, str):
                text = str(text)
            return (text.replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&apos;'))
        
        table_xml += f"""  <item priority="{opt['priority']}">
    <attr_name_and_direction>{escape_xml(opt['attr_name'])} - {escape_xml(opt['direction'])}</attr_name_and_direction>
    <current_performance>{escape_xml(opt['current_performance'])}</current_performance>
    <user_criteria>{escape_xml(opt['user_criteria'])}</user_criteria>
    <importance_score>{opt['importance_score']:.6f}</importance_score>
    <reference_suggestion>{escape_xml(opt['reference_suggestion'])}</reference_suggestion>
    <reference_user_id>{escape_xml(opt.get('reference_gid', '（无）'))}</reference_user_id>
    <reference_user_profile>{escape_xml(opt['reference_user_profile'])}</reference_user_profile>
    <reference_score>{escape_xml(opt['reference_score'])}</reference_score>
    <reference_valuation>{escape_xml(opt['reference_valuation'])}</reference_valuation>
  </item>
"""
    table_xml += "</optimization_priority_table>"
    
    user_prompt = f"""
<task>
你是一名产品经理，负责根据市场反馈的结果对产品进行优化，尽可能提高产品的市场回报率（购买产品的人数*产品的销售价格）。

请按照以下步骤完成优化任务：
1. 仔细阅读当前产品的完整设计文档
2. 理解产品的用户维度划分和属性体系
3. 分析属性优化优先顺序表，理解每个优化方向的意义和重要性
4. 思考如何采纳优化建议来改进产品设计，确保所有改动都来源于优化建议，不对其它部分进行多余改动
5. 确保优化的技术可行性：优化后的产品设计必须可生产且无矛盾
6. 生成优化后的产品设计文档
7. 为每个优化方向生成建议分析（是否采纳及原因）
8. 主动论证当前优化的可行性
</task>

<current_product_document>
<hard_design>
{hard_design}
</hard_design>
<core_features>
{core_features}
</core_features>
<value_proposition>
{value_proposition}
</value_proposition>
</current_product_document>

<user_dimensions>
{dims_info}
</user_dimensions>

<product_attributes>
{attrs_info}
</product_attributes>

<market_research_explanation>
根据产品的几个属性，我们找到了与之对应的一系列用户维度。在每个用户维度上分出对相关属性感知为抗拒（负价值）、不关心（无价值）、喜欢（正价值）的三部分，其中可能某一部分对应的用户不存在。之后对各种用户维度取值组合的用户群体选取代表性的消费者进行了调研，获取了他们对当前产品的估值（在产品售价为多少钱以下时愿意购买）、对产品各个属性的满意度打分以及对各个属性的优化建议。
</market_research_explanation>

{table_xml}

<optimization_table_explanation>
1. **三种优化方向**：三种优化方向分别关注该属性所对应用户维度的三种取值群体。
2. **当前表现**：优化前产品在该属性上的具体表现。
3. **用户判定标准**：该属性对应用户维度在该取值的判定标准。
4. **重要性分数**：根据市场反馈，计算对应群体所占的人群比例与价格对该维度的灵敏度相乘得到。取值越高说明优化该方向对产品销售表现的提高潜力越大。本表格以此为标准降序排序。
5. **参考优化建议及对应群体**：选择对应取值的各个群体中，估值最接近最佳定价的群体（最佳定价附近的群体的估值是决定产品售价的瓶颈群体，优先提高他们的估值对提高产品的整体市场表现最重要）。
6. **提供优化建议群体对该属性的打分**：打分的取值为-2、-1、0、1、2五种，分别对应非常抗拒、抗拒、无感、喜欢、非常喜欢。
</optimization_table_explanation>

<optimization_rules>
1. 尽可能多的采纳表中所有优化方向的所有优化建议。当优化建议之间有冲突时，优先采纳排在前面的建议。
2. 优化的唯一目的是提高各个用户群体对产品的估值，即愿意以更高的价格购买。
3. 技术可行性要求：必须确保优化后的产品设计可生产且无矛盾。所有优化必须考虑实际生产的技术限制和成本约束。
4. 改动限制：除根据采纳的建议进行针对性修改以外，不对其它部分进行多余改动。所有的改动必须来源于优化建议，不能随意添加或删除未在建议中提及的内容。
</optimization_rules>

<output_format>
输出文档必须包括以下部分，且每个部分用XML标签包裹。不要添加任何无关内容。

<hard_design>
优化后的硬性设计部分：包括技术指标、设计细节、材料规格、尺寸等具体设计元素。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只根据采纳的优化建议进行针对性修改，不对其它部分进行多余改动。
</hard_design>

<core_features>
优化后的核心功能介绍：描述产品的主要功能、优势、如何使用以及解决的用户痛点。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只根据采纳的优化建议进行针对性修改，不对其它部分进行多余改动。
</core_features>

<value_proposition>
优化后的价值定位部分：说明产品的市场定位、目标用户群、竞争优势和独特卖点。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只根据采纳的优化建议进行针对性修改，不对其它部分进行多余改动。
</value_proposition>

<feasibility_analysis>
优化可行性论证：主动论证当前优化的可行性，包括但不限于：
1. 技术可行性：说明优化后的设计在技术上是否可行，是否存在技术难点或限制
2. 生产可行性：说明优化后的设计是否可生产，生产过程中是否存在困难
3. 一致性检查：说明优化后的设计是否存在内部矛盾，各部分是否协调一致
4. 成本合理性：简要说明优化对成本的影响是否在可接受范围内
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</feasibility_analysis>

<suggestion_analysis>
<item attr_name="属性名称" direction="优化方向" adopted="是否采纳（true/false）">
采纳/不采纳该建议的原因说明，如果采纳了该建议，如何体现在优化后的设计文档上。
</item>
<item attr_name="属性名称" direction="优化方向" adopted="是否采纳（true/false）">
...
</item>
...
</suggestion_analysis>

注意：
1. 所有的输出内容必须使用中文
2. suggestion_analysis中的每个item必须与optimization_priority_table中的每一行一一对应
3. 属性名称和优化方向必须与表格中的"属性名称+优化方向"字段内容完全一致
4. 所有改动必须来源于优化建议，不能随意添加或删除未在建议中提及的内容
</output_format>
"""
    
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    
    reasoning_text, result_content = await call_reasoner(system_prompt, user_prompt, reasoning_effort="high", debug=False)
    logger(f"[{uniq_id}] LLM思考过程: \n{reasoning_text}")
    logger(f"[{uniq_id}] LLM输出原文: \n{result_content}")
    
    # 解析LLM输出
    hard_design_new = ""
    core_features_new = ""
    value_proposition_new = ""
    feasibility_analysis = ""
    suggestion_analysis = []
    
    hard_design_match = re.search(r'<hard_design>(.*?)</hard_design>', result_content, re.DOTALL)
    if hard_design_match:
        hard_design_new = hard_design_match.group(1).strip()
    else:
        raise ValueError("[硬性设计]未正确生成")
    
    core_features_match = re.search(r'<core_features>(.*?)</core_features>', result_content, re.DOTALL)
    if core_features_match:
        core_features_new = core_features_match.group(1).strip()
    else:
        raise ValueError("[核心功能]未正确生成")
    
    value_proposition_match = re.search(r'<value_proposition>(.*?)</value_proposition>', result_content, re.DOTALL)
    if value_proposition_match:
        value_proposition_new = value_proposition_match.group(1).strip()
    else:
        raise ValueError("[价值定位]未正确生成")
    
    feasibility_analysis_match = re.search(r'<feasibility_analysis>(.*?)</feasibility_analysis>', result_content, re.DOTALL)
    if feasibility_analysis_match:
        feasibility_analysis = feasibility_analysis_match.group(1).strip()
    else:
        raise ValueError("[优化可行性论证]未正确生成")
    
    suggestion_analysis_match = re.search(r'<suggestion_analysis>(.*?)</suggestion_analysis>', result_content, re.DOTALL)
    if suggestion_analysis_match:
        analysis_text = suggestion_analysis_match.group(1)
        # 解析每个item
        item_pattern = r'<item\s+attr_name="([^"]+)"\s+direction="([^"]+)"\s+adopted="([^"]+)"\s*>(.*?)</item>'
        items = re.findall(item_pattern, analysis_text, re.DOTALL)
        for attr_name, direction, adopted, content in items:
            suggestion_analysis.append({
                'attr_name': attr_name,
                'direction': direction,
                'adopted': adopted.lower() == 'true',
                'analysis': content.strip()
            })
    else:
        raise ValueError("[建议分析列表]未正确生成")
    
    # 验证建议分析列表与优化表一一对应
    if len(suggestion_analysis) != len(optimization_table):
        raise ValueError(f"建议分析列表项数({len(suggestion_analysis)})与优化优先顺序表项数({len(optimization_table)})不匹配")
    
    for i, (opt, analysis) in enumerate(zip(optimization_table, suggestion_analysis)):
        expected_key = f"{opt['attr_name']} - {opt['direction']}"
        actual_key = f"{analysis['attr_name']} - {analysis['direction']}"
        if expected_key != actual_key:
            raise ValueError(f"建议分析列表第{i+1}项的属性名称+优化方向不匹配：期望'{expected_key}'，实际'{actual_key}'")
    
    logger(f"[{uniq_id}] 产品优化完成")
    
    # 创建新行并更新字段
    new_row = row.copy()
    
    # 将原字段内容移动到_old字段
    new_row['hard_design_old'] = row.get('hard_design', '')
    new_row['core_features_old'] = row.get('core_features', '')
    new_row['value_proposition_old'] = row.get('value_proposition', '')
    
    # 替换为新字段内容
    new_row['hard_design'] = hard_design_new
    new_row['core_features'] = core_features_new
    new_row['value_proposition'] = value_proposition_new
    new_row['feasibility_analysis'] = feasibility_analysis
    new_row['suggestion_analysis'] = json.dumps(suggestion_analysis, ensure_ascii=False)
    
    return new_row


def filter_existing_optimization(product_row: pd.Series, existing_output_df: pd.DataFrame) -> bool:
    """
    判断某个产品是否需要优化。
    如果该产品的优化结果已存在于输出df中，则不需要处理。
    
    参数:
    - product_row: 产品行数据（Series），包含uniq_id（产品ID）和users_data
    - existing_output_df: 已存在的输出DataFrame（一行一个用户）
    
    返回:
    - True: 需要处理（不存在或数据不完整）
    - False: 不需要处理（已存在且完整）
    """
    product_uniq_id = product_row.get('uniq_id')  # 产品ID
    if product_uniq_id is None:
        # 如果没有产品ID，默认处理
        return True
    
    # 如果输出DataFrame为空，则所有产品都需要处理
    if existing_output_df is None or existing_output_df.empty:
        return True
    
    # 检查输出文件中是否存在suggestion_analysis字段
    if 'suggestion_analysis' not in existing_output_df.columns:
        return True
    
    # 从product_row中获取该产品的所有用户ID
    users_data = safe_json_loads(product_row.get('users_data', '[]'))
    if not isinstance(users_data, list) or len(users_data) == 0:
        return True
    
    user_uniq_ids = set()
    for user_data in users_data:
        user_uniq_id = user_data.get('user_uniq_id') or user_data.get('uniq_id')
        if user_uniq_id:
            user_uniq_ids.add(str(user_uniq_id))
    
    if not user_uniq_ids:
        return True
    
    # 检查输出文件中是否有该产品的任何用户行，且包含suggestion_analysis
    # 由于该产品的所有用户行都会有相同的suggestion_analysis，只需检查一个用户即可
    matching_rows = existing_output_df[existing_output_df['uniq_id'].isin(user_uniq_ids)]
    
    if matching_rows.empty:
        # 输出文件中没有该产品的用户行，需要处理
        return True
    
    # 检查是否有suggestion_analysis且不为空
    for _, row in matching_rows.iterrows():
        suggestion_analysis = row.get('suggestion_analysis')
        if pd.notna(suggestion_analysis) and str(suggestion_analysis).strip():
            # 找到了该产品的用户行且包含suggestion_analysis，说明已处理
            return False
    
    # 虽然有该产品的用户行，但没有suggestion_analysis，需要处理
    return True

