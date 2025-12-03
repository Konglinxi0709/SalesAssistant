import json
import os
from typing import Dict, List
import pandas as pd

def load_expenditure_categories() -> Dict[str, str]:
    """预加载年度消费支出统计与分类明细，供提示词注入。"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    annual_path = os.path.join(base_dir, "dataset", "supporting", "annual_consumption.csv")
    categories_path = os.path.join(base_dir, "dataset", "supporting", "consumption_expenditure_categories.csv")

    annual_df = pd.read_csv(annual_path)
    annual_lines = [f"- {row['category']}: {row['amount(Yuan)']} 元/人·年" for _, row in annual_df.iterrows()]
    annual_text = "全年中国城镇居民人均消费支出分类（单位：元/人·年）：\n" + "\n".join(annual_lines)

    categories_df = pd.read_csv(categories_path)
    category_sections: List[str] = []
    for first_level, grp in categories_df.groupby("first_level_name"):
        section_lines: List[str] = []
        for _, row in grp.iterrows():
            desc_value = row.get("description")
            description = str(desc_value).strip() if pd.notna(desc_value) else ""
            third_value = row.get("third_level_categories", "")
            third_entries: List[str] = []
            if isinstance(third_value, str) and third_value.strip():
                try:
                    third_items = json.loads(third_value)
                    if isinstance(third_items, list):
                        for item in third_items:
                            if isinstance(item, dict):
                                third_id = item.get("id", "").strip()
                                third_name = item.get("name", "").strip()
                                if third_id or third_name:
                                    third_entries.append(f"{third_id}-{third_name}".strip("-"))
                except json.JSONDecodeError:
                    third_entries = []
            detail_parts: List[str] = []
            if description:
                detail_parts.append(description)
            if third_entries:
                detail_parts.append("三级细分：" + "、".join(third_entries))
            detail_text = "；".join(detail_parts)
            section_lines.append(
                f"  - {row['second_level_name']}{('：' + detail_text) if detail_text else ''}"
            )
        category_sections.append(f"{first_level}：\n" + "\n".join(section_lines))

    categories_text = "国家统计的消费支出分类层级（按大类-小类）：\n" + "\n\n".join(category_sections)
    return {
        "annual_text": annual_text,
        "categories_text": categories_text,
    }