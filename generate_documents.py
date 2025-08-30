import csv
import re
import asyncio
from utils import call_reasoner

# 定义系统提示词，用于设置LLM的角色和任务
system_prompt = """你是一个产品设计专家，负责根据提供的产品信息生成详细的产品设计文档。你必须使用中文输出。"""

async def process_row(row):
    product_name = row['product_name']
    product_category_tree = row['product_category_tree']
    retail_price = row['retail_price']
    description = row['description']
    product_specifications = row['product_specifications']
    print(f"开始处理产品{product_name}")
    # 构建用户提示词，注入产品信息
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
  注意添加￥符号，且无任何其他内容
</cost_estimate>

<core_features>
  核心功能介绍：描述产品的主要功能、优势、如何使用以及解决的用户痛点。
  格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</core_features>

<value_proposition>
  价值定位部分：说明产品的市场定位、目标用户群、竞争优势和独特卖点。
  格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</value_proposition>
</output_format>
<product_info>
  <product_name>{product_name}</product_name>
  <product_category_tree>{product_category_tree}</product_category_tree>
  <retail_price>{retail_price}</retail_price>
  <description>{description}</description>
  <product_specifications>{product_specifications}</product_specifications>
</product_info>

"""

    try:
        result_content = await call_reasoner(system_prompt, user_prompt, debug=False)
    except Exception as e:
        print(f"Error calling LLM for product {product_name}: {e}")
        result_content = ""

    # 解析LLM输出，提取四个部分的内容
    modified_name = ""
    hard_design = ""
    cost_estimate = ""
    core_features = ""
    value_proposition = ""

    # 提取优化名称部分
    modified_name_match = re.search(r'<modified_name>(.*?)</modified_name>', result_content, re.DOTALL)
    if modified_name_match:
        modified_name = modified_name_match.group(1).strip()
    else:
        modified_name = "Not generated"

    # 提取硬性设计部分
    hard_design_match = re.search(r'<hard_design>(.*?)</hard_design>', result_content, re.DOTALL)
    if hard_design_match:
        hard_design = hard_design_match.group(1).strip()
    else:
        hard_design = "Not generated"

    # 提取成本估计部分
    cost_estimate_match = re.search(r'<cost_estimate>(.*?)</cost_estimate>', result_content, re.DOTALL)
    if cost_estimate_match:
        cost_estimate = cost_estimate_match.group(1).strip()
    else:
        cost_estimate = "Not generated"

    # 提取核心功能介绍部分
    core_features_match = re.search(r'<core_features>(.*?)</core_features>', result_content, re.DOTALL)
    if core_features_match:
        core_features = core_features_match.group(1).strip()
    else:
        core_features = "Not generated"

    # 提取价值定位部分
    value_proposition_match = re.search(r'<value_proposition>(.*?)</value_proposition>', result_content, re.DOTALL)
    if value_proposition_match:
        value_proposition = value_proposition_match.group(1).strip()
    else:
        value_proposition = "Not generated"

    # 更新行数据，添加新字段
    row['modified_name'] = modified_name
    row['hard_design'] = hard_design
    row['cost_estimate'] = cost_estimate
    row['core_features'] = core_features
    row['value_proposition'] = value_proposition
    print("="*100 + "\n" +
        f"**处理产品完成:{modified_name}**" + "\n" +
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
    return row

async def main():
    # 读取输入CSV
    with open('samples.csv', 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['modified_name', 'hard_design', 'cost_estimate', 'core_features', 'value_proposition']
        rows = []
        for row in reader:
            rows.append(row)

    # 并行处理每一行
    processed_rows = await asyncio.gather(*(process_row(row) for row in rows))

    # 写入输出CSV
    with open('product_documents.csv', 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in processed_rows:
            writer.writerow(row)

    print("Processing complete. Output written to product_documents.csv")

if __name__ == "__main__":
    asyncio.run(main())