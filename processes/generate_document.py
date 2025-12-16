import re
import pandas as pd # Added pandas import
from typing import Dict, Any, Callable
from utils.call_llm import call_reasoner, call_chat # Assuming call_reasoner is now in utils/call_llm.py

# 定义系统提示词，用于设置LLM的角色和任务
system_prompt = """你是一个产品设计专家，负责根据提供的产品信息生成详细的产品设计文档。你必须使用中文输出。"""

async def generate_document_task(row: Dict[str, Any], logger: Callable[[str], None]) -> Dict[str, Any]:
    """生成产品设计文档的任务回调函数。"""
    product_name = row.get('product_name', '')
    product_category_tree = row.get('product_category_tree', '')
    retail_price = row.get('retail_price', '')
    description = row.get('description', '')
    product_specifications = row.get('product_specifications', '')
    uniq_id = row.get('uniq_id', 'unknown_id') # Ensure uniq_id is available for logging

    logger(f"[{uniq_id}] 开始处理产品: {product_name}")

    user_prompt = f"""
<task>
请根据product_info中的产品信息，发挥想象，补充出产品的完整信息，并撰写一个产品设计文档。
</task>
<constraints>
1. 除成本估计部分外的任何其他部分不能出现与产品价格相关的任何数字，如成本、售价等！
</constraints>

<output_format>
输出文档必须包括以下四个部分，且每个部分用XML标签包裹。不要添加任何无关内容。
<modified_name>调整后的商品名。如果该产品的原名能准确概况产品信息，可以直接用于销售，直接翻译成中文即可。否则你需要重写一个商品名作为该产品销售时的名称。</modified_name>
<hard_design>
  硬性设计部分：包括技术指标、设计细节、材料规格、尺寸等具体设计元素。请基于现有信息推断并补充详细的设计参数。
  格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</hard_design>

<cost_estimate>
  成本估计部分：估计产品的生产成本、材料成本、劳动力成本等，并提供简要的估算依据。
  格式：总成本数字￥
  注意：1. 总成本数字需保留两位小数；2. 必须最后添加￥符号；3. 无任何其他内容
</cost_estimate>

<core_features>
  核心功能介绍：描述产品的主要功能、优势、如何使用以及解决的用户痛点。
  格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</core_features>

<value_proposition>
  价值定位部分：说明产品的市场定位、目标用户群、竞争优势和独特卖点。
  格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</value_proposition>
注意：所有的输出内容必须使用中文
</output_format>
<product_info>
  <product_name>{product_name}</product_name>
  <product_category_tree>{product_category_tree}</product_category_tree>
  <retail_price>{retail_price}</retail_price>
  <description>{description}</description>
  <product_specifications>{product_specifications}</product_specifications>
</product_info>

"""
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    result_content = await call_chat(system_prompt, user_prompt, debug=False)
    logger(f"[{uniq_id}] LLM输出原文: \n{result_content}")

    # 解析LLM输出，提取四个部分的内容
    modified_name = ""
    hard_design = ""
    cost_estimate = ""
    core_features = ""
    value_proposition = ""

    modified_name_match = re.search(r'<modified_name>(.*?)</modified_name>', result_content, re.DOTALL)
    if modified_name_match:
        modified_name = modified_name_match.group(1).strip()
    else:
        raise ValueError("[调整后产品名]未正确生成")

    hard_design_match = re.search(r'<hard_design>(.*?)</hard_design>', result_content, re.DOTALL)
    if hard_design_match:
        hard_design = hard_design_match.group(1).strip()
    else:
        raise ValueError("[硬性设计]未正确生成")

    cost_estimate_match = re.search(r'<cost_estimate>(.*?)</cost_estimate>', result_content, re.DOTALL)
    if cost_estimate_match:
        cost_estimate = cost_estimate_match.group(1).strip()
        # 判断cost_estimate格式是否为金额数字+￥
        if isinstance(cost_estimate, str) and cost_estimate.endswith('￥'):
            amount_str = cost_estimate[:-1].strip()
            try:
                amount = float(amount_str)
                cost_estimate = f"{amount:.2f}￥"
            except Exception:
                raise ValueError(f"[成本估计]格式不正确: '{cost_estimate}'，应为浮点数加￥结尾。")
        else:
            raise ValueError(f"[成本估计]格式不正确: '{cost_estimate}'，应为浮点数加￥结尾。")
            
    else:
        raise ValueError(f"[成本估计]未正确生成")

    core_features_match = re.search(r'<core_features>(.*?)</core_features>', result_content, re.DOTALL)
    if core_features_match:
        core_features = core_features_match.group(1).strip()
    else:
        raise ValueError("[核心功能]未正确生成")

    value_proposition_match = re.search(r'<value_proposition>(.*?)</value_proposition>', result_content, re.DOTALL)
    if value_proposition_match:
        value_proposition = value_proposition_match.group(1).strip()
    else:
        raise ValueError("[价值定位]未正确生成")

    # 更新行数据，添加新字段。使用一个新字典来避免直接修改传入的row
    output_row = row.copy()
    output_row['modified_name'] = modified_name
    output_row['hard_design'] = hard_design
    output_row['cost_estimate'] = cost_estimate
    output_row['core_features'] = core_features
    output_row['value_proposition'] = value_proposition
    
    logger("="*100 + "\n" +
        f"[{uniq_id}] **处理产品完成:{modified_name}**" + "\n" +
        "="*100 + "\n" +
        "【硬性设计】" + "\n" +
        hard_design + "\n" +
        "-"*100 + "\n" +
        f"【成本】:{cost_estimate}" + "\n" +
        "-"*100 + "\n" +
        "【核心功能】" + "\n" +
        core_features + "\n" +
        "-"*100 + "\n" +
        "【价值定位】" + "\n" +
        value_proposition + "\n" +
        "="*100
    )
    return output_row

def filter_existing_document(row: Dict[str, Any], existing_output_df: pd.DataFrame) -> bool:
    """判断某一行是否需要生成文档。如果uniq_id已存在于输出df中，则不需要生成。"""
    uniq_id = row.get('uniq_id')
    if uniq_id is None:
        # 如果没有uniq_id，默认处理它，或者根据需求选择跳过
        return True 
    
    # 如果输出DataFrame为空，则所有行都需要处理
    if existing_output_df.empty:
        return True

    # 检查uniq_id是否在现有输出文件的'uniq_id'列中
    return uniq_id not in existing_output_df['uniq_id'].values
