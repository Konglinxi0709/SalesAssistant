import csv
import json
from utils import simple_call_llm

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
    system_prompt = "你是一个商品分类助手。"
    user_prompt = f"""
<task>
1. 阅读商品类别列表和读取产品设计文档
2. 理解当前产品的属性，在商品类别列表中找到最符合的一类
3. 按要求输出
</task>
{categories_xml}
<product_document>
{product_document}
</product_document>
<output_format>
输出必须是JSON格式，只包含一个键"category"，值为最匹配的类别的名称（直接使用XML中的name字段值）
不要使用```json```或``````包裹
<output_format>
"""
    return system_prompt, user_prompt.strip()

def classify_product(product_document, csv_file_path="categories.csv"):
    """调用大模型对产品进行分类，返回类名字符串"""
    # 从CSV文件读取类别信息
    categories = read_categories_from_csv(csv_file_path)
    categories_xml = generate_categories_xml(categories)
    # 生成提示词
    system_prompt, user_prompt = generate_category_prompt(product_document, categories_xml)
    print("="*100)
    print("="*100)
    print(user_prompt)
    print("="*100)
    print("="*100)
    # 调用大模型
    output_text = simple_call_llm(system_prompt, user_prompt, debug=False)
    # 解析JSON输出
    try:
        output_json = json.loads(output_text)
        category_name = output_json["category"]
    except json.JSONDecodeError:
        category_name = "解析失败"
    
    return category_name

if __name__ == "__main__":

    input_file = "product_documents.csv"
    output_file = "classified_products.csv"

    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["category"] if "category" not in reader.fieldnames else reader.fieldnames
        for row in reader:
            # 组装产品设计文档
            product_name = row.get("modified_name", "")
            hard_design = row.get("hard_design", "")
            cost_estimate = row.get("cost_estimate", "")
            core_features = row.get("core_features", "")
            value_proposition = row.get("value_proposition", "")

            doc = (
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

            # 调用分类函数
            category = classify_product(doc)
            row["category"] = category
            rows.append(row)

    # 写入到新的CSV文件
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"已完成分类，结果已保存到 {output_file}")