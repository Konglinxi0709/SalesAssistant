import json
import os
from typing import Any, Callable, Dict, List

import pandas as pd

from utils.call_cuc_llm import call_reasoner, call_chat
from utils.load_expenditure_categories import load_expenditure_categories


SUPPORTING_TEXTS = load_expenditure_categories()

# 预加载年度消费支出数据
_base_dir = os.path.dirname(os.path.dirname(__file__))
_annual_consumption_path = os.path.join(_base_dir, "dataset", "supporting", "annual_consumption.csv")
_annual_consumption_df = pd.read_csv(_annual_consumption_path)
_ANNUAL_CONSUMPTION_DICT = dict(zip(_annual_consumption_df['category'], _annual_consumption_df['amount(Yuan)']))


async def consumer_feedback_task(row: Dict[str, Any], logger: Callable[[str], None]) -> Dict[str, Any]:
    """消费者反馈扮演任务回调函数。"""
    uniq_id = row.get("uniq_id", "unknown_id")
    product_name = row.get("modified_name", "")
    consumer_id = row.get("consumer_id", "")

    hard_design = str(row.get("hard_design", "") or "")
    core_features = str(row.get("core_features", "") or "")
    value_proposition = str(row.get("value_proposition", "") or "")
    consumer_definition = str(row.get("consumer_definition", "") or "")
    
    # 从segmentation_result中提取产品属性列表
    segmentation_result_str = row.get("segmentation_result", "")
    product_attributes: List[Dict[str, Any]] = []
    filtered_attributes: List[Dict[str, Any]] = []
    if segmentation_result_str and pd.notna(segmentation_result_str):
        try:
            segmentation_data = json.loads(segmentation_result_str)
            total_attributes = segmentation_data.get("total_attributes", [])
            segmentations = segmentation_data.get("segmentations", [])

            # 原始属性列表
            if isinstance(total_attributes, list):
                product_attributes = total_attributes

            # 仅保留“被分配到某个用户维度”的属性
            assigned_attr_names: List[str] = []
            if isinstance(segmentations, list):
                for seg in segmentations:
                    if not isinstance(seg, dict):
                        continue
                    pos = seg.get("positive_attribute_names", []) or []
                    neg = seg.get("negative_attribute_names", []) or []
                    for name in list(pos) + list(neg):
                        if isinstance(name, str) and name not in assigned_attr_names:
                            assigned_attr_names.append(name)

            if product_attributes and assigned_attr_names:
                filtered_attributes = [
                    attr for attr in product_attributes
                    if isinstance(attr, dict) and attr.get("attribute_name", "") in assigned_attr_names
                ]
        except (json.JSONDecodeError, KeyError) as e:
            logger(f"[{uniq_id}] 警告: 无法从segmentation_result中提取属性列表: {e}")
    
    if not product_attributes:
        raise ValueError(f"[消费者反馈]无法获取产品属性列表，请确保segmentation_result中包含total_attributes字段")

    if filtered_attributes:
        product_attributes = filtered_attributes
    else:
        # 如果没有任何属性被分配到用户维度，则直接报错提示
        raise ValueError(f"[消费者反馈]{uniq_id} 未找到分配到用户维度的属性，请检查segmentation_result的segmentations中的positive/negative_attribute_names")
    
    # 获取消费支出分类和对应的年度预算
    expenditure_category = str(row.get("expenditure_category", "") or "").strip()
    annual_budget = None
    if expenditure_category:
        annual_budget = _ANNUAL_CONSUMPTION_DICT.get(expenditure_category)
        if annual_budget is None:
            logger(f"[{uniq_id}] 警告: 未找到消费类别 '{expenditure_category}' 对应的年度支出数据")
    else:
        logger(f"[{uniq_id}] 提示: 该产品不属于居民消费支出范畴")

    logger(f"[{uniq_id}] 开始消费者反馈扮演: {product_name} / {consumer_id}，共 {len(product_attributes)} 个已分配属性")

    system_prompt = (
        "你现在就是指定消费者本人。你将始终使用第一人称“我”来表达，不能以旁观者视角或第三人称分析自己。"
        "你具备严谨的自我反思能力，能够依照给定资料、真实情境与逻辑推导出可信的画像、价值评估、心理价格与优化建议。"
    )

    # 构建消费预算信息
    budget_info = ""
    if expenditure_category and annual_budget is not None:
        budget_info = f"""
<consumption_budget>
当前产品所属消费支出类别：{expenditure_category}
我对此类产品/服务的全年消费预算：{annual_budget} 元/人·年

注意：这是我在该消费类别下的全年总预算，需要覆盖该类别下的所有必需消费品和服务。在评估当前产品的心理价格时，我需要有节制地考虑该产品在此预算中的合理占比，不能将全部预算用于单一产品。
</consumption_budget>
"""
    else:
        budget_info = """
<consumption_budget>
当前产品不属于居民消费支出范畴，无对应的消费预算基准。
</consumption_budget>
"""

    # 构建产品属性信息
    attributes_info = ""
    for idx, attr in enumerate(product_attributes, 1):
        attr_name = attr.get("attribute_name", "")
        attr_desc = attr.get("attribute_description", "")
        attributes_info += f"属性 {idx}: {attr_name}\n  描述: {attr_desc}\n\n"
    
    user_prompt = f"""
<task>
你要依据consumer_group_definition构建"我"的用户画像，并完成后续分析。consumer_group_definition包含：
1. potential_user_group：定义"我"必须属于的潜在用户群体整体，是构建画像时必须契合的基础条件。
2. 各个Dimension：每个Dimension的内容是"属于该部分的条件"，表示"我"在该维度上必须满足的条件，也是构建画像时必须契合的。

你必须确保"我"的画像完全满足potential_user_group和所有Dimension中标注的条件，然后完成如下步骤，并全程使用第一人称"我"来描述：
1) 你要构建一个唯一且具代表性的【用户画像】，确认我满足全部群体判定标准（包括potential_user_group和所有Dimension的条件）；
2) 你要输出一个【用户画像合理性论证】，分条地充分证明当前的用户画像满足potential_user_group的定义要求，以及每个Dimension上具体判定标准的定义要求；
3) 你要针对product_attributes中列出的每个属性，分析该属性在当前产品上的表现在我的实际使用中会带来什么样的作用效果，会给我造成麻烦或困扰，还是能解决我的实际问题、提高使用体验；
4) 你要对每个属性进行价值打分（-2到2）
5) 你要为每个属性提出针对性的优化建议，说明如何改进该属性以提升我的心理价格。如果某个属性你认为已经无需优化，可以不提建议。
6) 你要综合所有属性的效用考虑，给出一个【心理价格】（当售价小于等于该价格时我愿意购买，否则不愿意）。
</task>

<constraints>
【对consumer_group_definition的理解与映射】
1. consumer_group_definition包含两部分：
   a) potential_user_group：定义"我"必须属于的潜在用户群体整体
   b) 各个Dimension：每个Dimension的内容是"属于该部分的条件"，即"我"必须满足的条件
2. 你必须逐条解析consumer_group_definition的全部条件，将每条定义要点映射为"我"的具体、可核查事实，做到一一对应、无遗漏；

【用户画像的评判标准】
1. 你要覆盖影响购买需求与使用体验的关键信息：身份背景、生活结构与节律、常见环境与工具、能力与心智特征、社交关系与角色、信息来源与信任偏好、与产品相关的典型场景与限制、相关行为的频次与时长、可支配资源与关键约束等；
2. 你陈述的每一项画像信息都必须唯一且具体，禁止使用范围性/选择性/概率性表述（不得出现"在…到…之间""是…或…""可能/大概/之一/介于…之间/视情况而定"等字样）；
3. 画像信息要自洽无冲突，并确保"我"无争议地满足consumer_group_definition全部判定标准，同时在现实生活中合理可信。

【用户画像合理性论证的评判标准】
1. 你必须在用户画像合理性论证中，分条地充分证明当前的用户画像满足以下要求：
   a) 证明"我"满足potential_user_group中定义的所有条件；
   b) 对于每个Dimension，证明"我"满足该Dimension中"属于该部分的条件"；
2. 合理性论证必须条理清晰，合理性一目了然，这必须建立在用户画像本身确实符合这些条件的基础上。

【属性分析的评判标准】
1. 你要针对product_attributes中的每个属性，基于我的真实使用情境，分析该属性在当前产品上的表现会给我带来什么样的作用效果（包括正面效果和负面效果）；
2. 你要明确说明每个属性的作用效果是通过什么机制产生的，在什么情境下发生，与我的画像要素（身份背景、生活场景、能力特征等）如何关联；
3. 你要基于作用效果的分析，对每个属性给出准确的价值打分（-2到2），其中-2代表强烈负价值(该属性的存在让我对当前产品的心理价格严重降低)，-1代表负价值(该属性的存在让我对当前产品的心理价格小幅降低)，0代表无影响，1代表正价值(该属性的存在让我对该产品的心理价格小幅提高)，2代表强烈正价值(该属性的存在让我对该产品的心理价格显著提高)。打分必须与作用效果分析在逻辑上一致；
4. 你要为每个属性提出针对性的优化建议，说明如何改进该属性以提升我的心理价格，建议要具体、可执行。**注意不能提降低产品成本、降低售价之类的建议。**
5. 如果你认为某个属性已经无需优化，可以不提建议，写为空字符串""即可。

【心理价格的评判标准】
1. 你只能依据所有属性的价值打分、consumption_budget中给出的预算约束与其它约束条件做出心理价格判断，禁止引用任何外部售价、成本、折扣、竞品或市场行情；
2. 如果产品属于居民消费支出范畴，你必须考虑该类支出的全年预算以及该类支出包含的所有必需消费品和服务，有节制地评估当前产品在该预算中的合理占比；
3. 心理价格必须以人民币金额数字表示（不含任何单位与符号），为纯浮点数且保留两位小数；该数值需能由所有属性的价值打分与约束条件推导并解释，可复核。
4. 你所有表述（包括列表条目）都必须使用第一人称"我"。
</constraints>

{budget_info}

<consumption_expenditure_categories>
{SUPPORTING_TEXTS["categories_text"]}

注意：以上是居民消费支出的完整分类层级。如果当前产品属于居民消费支出范畴，你需要参考该类别下的所有细分项目，理解该类支出包含的必需消费品和服务范围，从而有节制地评估当前产品在该类别预算中的合理占比。
</consumption_expenditure_categories>

<product_info>
<modified_name>{product_name}</modified_name>
<hard_design>{hard_design}</hard_design>
<core_features>{core_features}</core_features>
<value_proposition>{value_proposition}</value_proposition>
</product_info>

<product_attributes>
{attributes_info}
</product_attributes>

{consumer_definition}

<thinking_guidance>
A. 你要先理解consumer_group_definition的结构，并将所有这些条件逐条转译为"我"的具体事实清单，确保完备与无冲突
B. 你要结合consumption_budget中给出的全年预算和consumption_expenditure_categories中该类别的完整细分，理解该类支出包含的所有必需消费品和服务，评估当前产品在该类别预算中的合理占比；
C. 在事实清单与预算约束的基础上，针对product_attributes中的每个属性，构造该属性在我实际使用中的真实情境链（起点/过程/结果），分析该属性在当前产品上的表现会给我带来什么样的作用效果；
D. 基于每个属性的作用效果分析，给出准确的价值打分（-2到2），并针对每个属性提出优化建议；
E. 综合所有属性的价值打分、预算约束与关键限制，推导心理价格的合理区间并定点，说明各属性的价值打分如何共同约束价格上限；
F. 保持全链路可回溯与可解释，确保每个属性的分析、打分和建议都能与画像要素和情境链对应。
</thinking_guidance>

<output_format>
输出必须为JSON，字段如下：
{{
  "user_profile": "详尽的具体用户画像",
  "user_profile_rationale": "用户画像合理性论证，分条地充分证明当前的用户画像满足potential_user_group的定义要求，以及每个Dimension上具体判定标准的定义要求",
  "attribute_analysis": [
    {{
      "attribute_name": "属性名称",
      "attribute_effect": "该属性对自己使用体验的作用效果的分析",
      "attribute_score": -2,
      "attribute_optimization_suggestion": "对该属性的优化建议，可以为空"
    }}
  ],
  "psychological_price": 10.00
}}
注意：
1. user_profile_rationale必须分条地充分证明用户画像满足potential_user_group和所有Dimension的判定标准，每条论证都要明确指出画像中的具体事实与对应条件的映射关系
2. attribute_analysis数组中的元素必须与product_attributes中的属性一一对应，且顺序一致
3. attribute_score必须是整数，取值范围为-2, -1, 0, 1, 2
4. 不要使用```json```或``````包裹
5. 所有的输出必须使用中文
</output_format>
"""

    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")

    reasoning_text, output_text = await call_reasoner(system_prompt, user_prompt, reasoning_effort="low", debug=False)
    # output_text = await call_chat(system_prompt, user_prompt, debug=False)
    # reasoning_text = ""
    logger(f"[{uniq_id}] LLM思考过程: \n{reasoning_text}")
    logger(f"[{uniq_id}] LLM输出原文: \n{output_text}")

    try:
        output_json = json.loads(output_text)
        required_fields = [
            "user_profile",
            "user_profile_rationale",
            "attribute_analysis",
            "psychological_price",
        ]
        for field in required_fields:
            if field not in output_json:
                raise ValueError(f"输出缺少字段: {field}")

        if not isinstance(output_json["user_profile"], str) or not output_json["user_profile"].strip():
            raise ValueError("user_profile 应为非空字符串")
        
        if not isinstance(output_json["user_profile_rationale"], str) or not output_json["user_profile_rationale"].strip():
            raise ValueError("user_profile_rationale 应为非空字符串")

        # 验证attribute_analysis
        attribute_analysis = output_json["attribute_analysis"]
        if not isinstance(attribute_analysis, list):
            raise ValueError("attribute_analysis 必须是数组")
        
        if len(attribute_analysis) != len(product_attributes):
            raise ValueError(f"attribute_analysis数组长度 {len(attribute_analysis)} 与产品属性数量 {len(product_attributes)} 不匹配")
        
        # 验证每个属性分析项，并检查与product_attributes一一对应
        expected_attr_names = [attr.get("attribute_name", "") for attr in product_attributes]
        for idx, analysis_item in enumerate(attribute_analysis):
            if not isinstance(analysis_item, dict):
                raise ValueError(f"attribute_analysis[{idx}] 必须是字典类型")
            
            required_attr_fields = ["attribute_name", "attribute_effect", "attribute_score", "attribute_optimization_suggestion"]
            for field in required_attr_fields:
                if field not in analysis_item:
                    raise ValueError(f"attribute_analysis[{idx}] 缺少字段: {field}")
            
            # 验证属性名称与product_attributes一一对应
            attr_name = analysis_item["attribute_name"]
            if idx < len(expected_attr_names):
                expected_name = expected_attr_names[idx]
                if attr_name != expected_name:
                    raise ValueError(f"attribute_analysis[{idx}].attribute_name '{attr_name}' 与预期属性名称 '{expected_name}' 不匹配")
            
            # 验证attribute_effect
            if not isinstance(analysis_item["attribute_effect"], str) or not analysis_item["attribute_effect"].strip():
                raise ValueError(f"attribute_analysis[{idx}].attribute_effect 应为非空字符串")
            
            # 验证attribute_score
            score = analysis_item["attribute_score"]
            if not isinstance(score, int) or score not in [-2, -1, 0, 1, 2]:
                raise ValueError(f"attribute_analysis[{idx}].attribute_score 必须是整数，且取值范围为-2, -1, 0, 1, 2，当前值为: {score}")
            
            # # 验证attribute_optimization_suggestion
            # if not isinstance(analysis_item["attribute_optimization_suggestion"], str) or not analysis_item["attribute_optimization_suggestion"].strip():
            #     raise ValueError(f"attribute_analysis[{idx}].attribute_optimization_suggestion 应为非空字符串")

        # 验证psychological_price
        price = output_json["psychological_price"]
        if not isinstance(price, (int, float)):
            raise ValueError("psychological_price 必须是数字")
        output_json["psychological_price"] = float(f"{float(price):.2f}")

        logger(f"[{uniq_id}] 消费者反馈生成成功")
    except json.JSONDecodeError as exc:
        logger(f"[{uniq_id}] JSON解析失败: {exc}")
        raise ValueError(f"[消费者反馈]JSON解析失败: {exc}")
    except Exception as exc:
        logger(f"[{uniq_id}] 消费者反馈生成失败: {exc}")
        raise

    new_row = row.copy()
    new_row["user_profile"] = output_json["user_profile"]
    new_row["user_profile_rationale"] = output_json["user_profile_rationale"]
    new_row["attribute_analysis"] = json.dumps(output_json["attribute_analysis"], ensure_ascii=False)
    new_row["psychological_price"] = output_json["psychological_price"]

    return new_row


def _list_from_entry(entry: Any) -> List[str]:
    """将现有字段（可能是JSON字符串或列表）规范化为列表。"""
    if isinstance(entry, list):
        return entry
    if isinstance(entry, str):
        entry = entry.strip()
        if not entry:
            return []
        try:
            parsed = json.loads(entry)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def filter_existing_feedback(row: Dict[str, Any], existing_output_df: pd.DataFrame) -> bool:
    """若已存在完整反馈，则跳过。"""
    uniq_id = row.get("uniq_id")
    consumer_id = row.get("consumer_id")
    if uniq_id is None or consumer_id is None:
        return True

    if existing_output_df is None or existing_output_df.empty:
        return True

    if "uniq_id" not in existing_output_df.columns or "consumer_id" not in existing_output_df.columns:
        return True

    matched = existing_output_df[
        (existing_output_df["uniq_id"] == uniq_id)
        & (existing_output_df["consumer_id"] == consumer_id)
    ]
    if matched.empty:
        return True

    row0 = matched.iloc[0]
    required_fields = [
        "user_profile",
        "user_profile_rationale",
        "attribute_analysis",
        "psychological_price",
    ]
    for field in required_fields:
        if field not in matched.columns:
            return True
        value = row0[field]
        if field == "attribute_analysis":
            # 验证attribute_analysis是否为有效的JSON数组
            if pd.isna(value) or str(value).strip() == "":
                return True
            try:
                parsed = json.loads(value) if isinstance(value, str) else value
                if not isinstance(parsed, list) or len(parsed) == 0:
                    return True
            except (json.JSONDecodeError, TypeError):
                return True
        else:
            if pd.isna(value) or str(value).strip() == "":
                return True

    return False
