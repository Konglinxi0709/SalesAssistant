import csv
import json
import pandas as pd
from typing import Dict, Any, Callable
from utils.call_llm import call_chat
from utils.load_expenditure_categories import load_expenditure_categories

# 预加载消费支出分类标准
EXPENDITURE_CATEGORIES = load_expenditure_categories()

def read_categories_from_csv(file_path):
    """从CSV文件中读取类别信息"""
    categories = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            categories.append(row)
    return categories

def generate_categories_xml(categories):
    """生成XML格式的类别字符串"""
    categories_xml = "<categories>\n"
    for category in categories:
        categories_xml += f"  <category>\n    <index>{category['index']}</index>\n    <explanation>{category['explanation']}</explanation>\n    <name>{category['name']}</name>\n  </category>\n"
    categories_xml += "</categories>"
    return categories_xml

def generate_category_prompt(product_document, categories_xml):
    """生成XML风格的提示词，其中包含类别信息和产品文档"""
    system_prompt = "你是一个商品分类助手，擅长对产品进行多维度分类。"
    
    expenditure_categories_text = EXPENDITURE_CATEGORIES.get("categories_text", "")
    
    user_prompt = f"""
<task>
你需要对当前产品进行两类分类：
1. 产品分类：在商品类别列表中找到最符合的一类
2. 消费支出分类：判断该产品是否属于居民消费支出范畴，如果属于，确定其所属的一级大类；如果不属于（例如工业原料、生产资料等非居民消费品），则输出空字符串
</task>

<product_categories>
{categories_xml}
</product_categories>

<consumption_expenditure_categories>
{expenditure_categories_text}
</consumption_expenditure_categories>

<product_document>
{product_document}
</product_document>

<classification_guidance>
1. 产品分类：仔细阅读产品设计文档，理解产品的核心属性、功能和使用场景，在商品类别列表中找到最匹配的类别名称
2. 消费支出分类：
   - 判断该产品是否为居民日常消费中会购买和使用的商品或服务
   - 如果是，根据消费支出分类层级，确定其所属的一级大类，包括：“食品烟酒”、“衣着”、“居住”、“生活用品及服务”、“交通和通信”、“教育、文化和娱乐”、“医疗保健”、“其他用品和服务”
   - 如果不是（例如：工业原料、生产设备、B2B产品、非消费品等），将expenditure_category字段输出为空字符串""
   - 注意：消费支出分类关注的是居民个人和家庭的消费行为，不包括企业采购、生产用途等非消费场景
</classification_guidance>

<output_format>
输出必须是JSON格式，包含两个字段：
{{
  "category": "产品分类名称（直接使用商品类别XML中的name字段值）",
  "expenditure_category": "消费支出一级大类名称（如：食品烟酒、衣着等），如果不属于居民消费支出则输出空字符串\"\""
}}
注意：
1. 不要使用```json```或``````包裹
2. 所有的输出必须使用中文
</output_format>
"""
    return system_prompt, user_prompt.strip()

async def classify_category_task(row: Dict[str, Any], logger: Callable[[str], None]) -> Dict[str, Any]:
    """分类产品的任务回调函数。"""
    uniq_id = row.get('uniq_id', 'unknown_id')
    product_name = row.get('modified_name', '')
    
    logger(f"[{uniq_id}] 开始分类产品: {product_name}")
    
    # 组装产品设计文档
    hard_design = row.get("hard_design", "")
    cost_estimate = row.get("cost_estimate", "")
    core_features = row.get("core_features", "")
    value_proposition = row.get("value_proposition", "")

    product_document = (
        "="*100 + "\n"
        + f"【产品名称】:{product_name}\n"
        + "="*100 + "\n"
        + "【硬性设计】\n"
        + str(hard_design) + "\n"
        + "-"*100 + "\n"
        + f"【成本】:{cost_estimate}\n"
        + "-"*100 + "\n"
        + "【核心功能】\n"
        + str(core_features) + "\n"
        + "-"*100 + "\n"
        + "【价值定位】\n"
        + str(value_proposition) + "\n"
        + "="*100
    )
    
    logger(f"[{uniq_id}] 产品文档组装完成")
    
    # 从CSV文件读取类别信息
    categories = read_categories_from_csv("dataset/supporting/product_categories.csv")
    categories_xml = generate_categories_xml(categories)
    
    # 生成提示词
    system_prompt, user_prompt = generate_category_prompt(product_document, categories_xml)
    
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    
    # 调用大模型
    output_text = await call_chat(system_prompt, user_prompt, debug=False)
    logger(f"[{uniq_id}] LLM输出原文: \n{output_text}")
    
    # 解析JSON输出
    try:
        output_json = json.loads(output_text)
        
        # 验证必需字段
        if "category" not in output_json:
            raise ValueError("JSON缺少category字段")
        if "expenditure_category" not in output_json:
            raise ValueError("JSON缺少expenditure_category字段")
        
        category_name = output_json["category"]
        expenditure_category = output_json["expenditure_category"]
        
        # 验证字段类型
        if not isinstance(category_name, str) or not category_name.strip():
            raise ValueError("category字段必须是非空字符串")
        if not isinstance(expenditure_category, str):
            raise ValueError("expenditure_category字段必须是字符串（可以为空）")
        
        # 如果expenditure_category不为空，验证其是否为有效的一级大类
        valid_expenditure_categories = [
            "食品烟酒", "衣着", "居住", "生活用品及服务",
            "交通和通信", "教育、文化和娱乐", "医疗保健", "其他用品和服务"
        ]
        if expenditure_category and expenditure_category not in valid_expenditure_categories:
            raise ValueError(f"expenditure_category '{expenditure_category}' 不在有效的一级大类列表中")
        
        logger(f"[{uniq_id}] 分类成功: 产品分类={category_name}, 消费支出分类={expenditure_category if expenditure_category else '(空)'}")
        
    except json.JSONDecodeError as e:
        logger(f"[{uniq_id}] JSON解析失败: {str(e)}")
        raise ValueError(f"[分类]JSON解析失败: {str(e)}")
    except Exception as e:
        logger(f"[{uniq_id}] 分类处理失败: {str(e)}")
        raise ValueError(f"[分类]处理失败: {str(e)}")
    
    # 更新行数据，添加新字段
    output_row = row.copy()
    output_row['category'] = category_name
    output_row['expenditure_category'] = expenditure_category
    
    logger(f"[{uniq_id}] **分类完成: {category_name}, 消费支出分类: {expenditure_category if expenditure_category else '(空)'}**")
    return output_row

def filter_existing_categories(row: Dict[str, Any], existing_output_df: pd.DataFrame) -> bool:
    """判断某一行是否需要分类。如果uniq_id已存在于输出df中且已有category和expenditure_category字段，则不需要分类。"""
    uniq_id = row.get('uniq_id')
    if uniq_id is None:
        # 如果没有uniq_id，默认处理它
        return True 
    
    # 如果输出DataFrame为空，则所有行都需要处理
    if existing_output_df.empty:
        return True

    # 检查uniq_id是否在现有输出文件中，且是否已有两个分类字段
    if uniq_id in existing_output_df['uniq_id'].values:
        # 找到对应的行
        matching_rows = existing_output_df[existing_output_df['uniq_id'] == uniq_id]
        if not matching_rows.empty:
            row0 = matching_rows.iloc[0]
            # 检查是否已有category字段且不为空
            has_category = 'category' in matching_rows.columns and pd.notna(row0.get('category')) and str(row0.get('category', '')).strip() != ''
            # 检查是否已有expenditure_category字段（可以为空字符串，但字段必须存在）
            has_expenditure_category = 'expenditure_category' in matching_rows.columns and pd.notna(row0.get('expenditure_category'))
            
            if has_category and has_expenditure_category:
                return False  # 已有完整分类，不需要重新分类
    
    return True  # 需要分类
