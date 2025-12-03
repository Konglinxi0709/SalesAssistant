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
    
    # 获取消费支出分类和对应的年度预算
    expenditure_category = str(row.get("expenditure_category", "") or "").strip()
    annual_budget = None
    if expenditure_category:
        annual_budget = _ANNUAL_CONSUMPTION_DICT.get(expenditure_category)
        if annual_budget is None:
            logger(f"[{uniq_id}] 警告: 未找到消费类别 '{expenditure_category}' 对应的年度支出数据")
    else:
        logger(f"[{uniq_id}] 提示: 该产品不属于居民消费支出范畴")

    logger(f"[{uniq_id}] 开始消费者反馈扮演: {product_name} / {consumer_id}")

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

    user_prompt = f"""
<task>
你要依据consumer_group_definition中“潜在用户群体的定义”与“该群体中当前所处部分的各个分类维度的判定”，严格确认“我”确实属于该群体。
在此基础上，你必须完成如下步骤，并全程使用第一人称“我”来描述：
1) 你要构建一个唯一且具代表性的【用户画像】，确认我满足全部群体判定标准；
2) 你要代入我的真实生活/学习/工作情境，枚举所有【正向价值】与【负向价值】；
3) 你要据此给出一个【心理价格】（当售价小于等于该价格时我愿意购买，否则不愿意）；
4) 你要提出唯一的【优化建议】，解释其如何提升我的心理价格。
</task>

<constraints>
【对consumer_group_definition的理解与映射】
1. 你必须逐条解析consumer_group_definition的全部边界条件、必要条件与充分条件；
2. 你要将每条定义要点映射为“我”的具体、可核查事实，做到一一对应、无遗漏；
3. 若定义包含排他/边界/阈值条件，你要在画像中以明确事实体现，杜绝含糊表述。

【用户画像的评判标准】
4. 你要覆盖影响购买需求与使用体验的关键信息：身份背景、生活结构与节律、常见环境与工具、能力与心智特征、社交关系与角色、信息来源与信任偏好、与产品相关的典型场景与限制、相关行为的频次与时长、可支配资源与关键约束等；
5. 你陈述的每一项画像信息都必须唯一且具体，禁止使用范围性/选择性/概率性表述（不得出现“在…到…之间”“是…或…”“可能/大概/之一/介于…之间/视情况而定”等字样）；
6. 画像信息要自洽无冲突，并确保“我”无争议地满足consumer_group_definition全部判定标准，同时在现实生活中合理可信。

【价值想象的评判标准】
7. 你要基于真实情境链列举每一项正向/负向价值，逐一说明触发情境、必要前提、边界条件、作用路径（通过什么机制影响什么结果）以及与画像要素的紧密关联，避免空泛或不可检验表述；
8. 你要覆盖效率/时间、精力/心智负担、舒适/健康与安全、社交与自我表达、可替代性与锁定、学习曲线与运维成本等主要维度（如适用），并在具体场景下形成可复核论证；
9. 你要为每一项价值指明在哪些条件下不成立或反向成立，使价值具备可否定性与明确边界。

【心理价格的评判标准】
10. 你只能依据前述价值权衡、consumption_budget中给出的预算约束与其它约束条件做出心理价格判断，禁止引用任何外部售价、成本、折扣、竞品或市场行情；
11. 如果产品属于居民消费支出范畴，你必须考虑该类支出的全年预算以及该类支出包含的所有必需消费品和服务，有节制地评估当前产品在该预算中的合理占比；
12. 心理价格必须以人民币金额数字表示（不含任何单位与符号），为纯浮点数且保留两位小数；该数值需能由价值路径与约束条件推导并解释，可复核。

【优化建议的评判标准】
13. 你必须只提出一条针对自身最关键价值驱动或主要阻碍的改进建议，该建议要具体、可执行、可落地，避免口号式表达；
14. 你要明确建议的作用路径：如何改变关键前提、如何改变价值路径以提升心理价格；建议必须与画像中的特征、情境限制和能力边界严格对齐；
15. 你所有表述（包括列表条目）都必须使用第一人称“我”。
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

<consumer_group_definition>
{consumer_definition}
</consumer_group_definition>

<thinking_guidance>
A. 你要先将consumer_group_definition逐要点转译为“我”的具体事实清单，确保完备与无冲突；
B. 你要结合consumption_budget中给出的全年预算和consumption_expenditure_categories中该类别的完整细分，理解该类支出包含的所有必需消费品和服务，评估当前产品在该类别预算中的合理占比；
C. 在事实清单与预算约束的基础上，构造若干高频真实使用情境链（起点/过程/结果），并标注触发条件、门槛、资源与限制；
D. 沿情境链识别全部正向与负向价值，明确机制路径、边界及与画像要素的强关联，形成可复核的价值集合与相对权重；
E. 基于价值集合、预算约束与关键限制，推导心理价格的合理区间并定点，说明主导价值、受限要素与关键阻碍如何共同约束价格上限；
F. 锁定最主要瓶颈提出唯一优化建议，论证其可实施性与提升逻辑，保持全链路可回溯与可解释。
</thinking_guidance>

<output_format>
输出必须为JSON，字段如下：
{{
  "user_profile": "详尽的具体用户画像",
  "positive_values": ["第一条正向价值", "第二条正向价值", ...],
  "negative_values": ["第一条负向价值", "第二条负向价值", ...],
  "psychological_price": 10.00,
  "optimization_suggestion": "最能提升其心理价格的一条优化建议"
}}
不要使用```json```或``````包裹
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
            "positive_values",
            "negative_values",
            "psychological_price",
            "optimization_suggestion",
        ]
        for field in required_fields:
            if field not in output_json:
                raise ValueError(f"输出缺少字段: {field}")

        if not isinstance(output_json["user_profile"], str) or not output_json["user_profile"].strip():
            raise ValueError("user_profile 应为非空字符串")

        for list_name in ["positive_values", "negative_values"]:
            values = output_json[list_name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{list_name} 应为非空列表")
            for idx, item in enumerate(values):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{list_name}[{idx}] 应为非空字符串")

        price = output_json["psychological_price"]
        if not isinstance(price, (int, float)):
            raise ValueError("psychological_price 必须是数字")
        output_json["psychological_price"] = float(f"{float(price):.2f}")

        if not isinstance(output_json["optimization_suggestion"], str) or not output_json["optimization_suggestion"].strip():
            raise ValueError("optimization_suggestion 应为非空字符串")

        logger(f"[{uniq_id}] 消费者反馈生成成功")
    except json.JSONDecodeError as exc:
        logger(f"[{uniq_id}] JSON解析失败: {exc}")
        raise ValueError(f"[消费者反馈]JSON解析失败: {exc}")
    except Exception as exc:
        logger(f"[{uniq_id}] 消费者反馈生成失败: {exc}")
        raise

    new_row = row.copy()
    new_row["user_profile"] = output_json["user_profile"]
    new_row["positive_values"] = json.dumps(output_json["positive_values"], ensure_ascii=False)
    new_row["negative_values"] = json.dumps(output_json["negative_values"], ensure_ascii=False)
    new_row["psychological_price"] = output_json["psychological_price"]
    new_row["optimization_suggestion"] = output_json["optimization_suggestion"]

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
        "positive_values",
        "negative_values",
        "psychological_price",
        "optimization_suggestion",
    ]
    for field in required_fields:
        if field not in matched.columns:
            return True
        value = row0[field]
        if field in ("positive_values", "negative_values"):
            if not _list_from_entry(value):
                return True
        else:
            if pd.isna(value) or str(value).strip() == "":
                return True

    return False
