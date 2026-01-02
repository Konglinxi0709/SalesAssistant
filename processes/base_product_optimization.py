import json
import re
import os
import pandas as pd
import asyncio
from typing import Any, Callable, Dict, List
from utils.call_cuc_llm import call_reasoner


def safe_json_loads(x):
    """安全地解析JSON字符串"""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    return x


async def base_product_optimization_task(row: pd.Series, logger: Callable[[str], None], reasoning_effort: str = "low") -> pd.Series:
    """
    基础产品优化任务：仅基于产品文档进行优化，不使用市场反馈结果
    
    参数:
    - row: 产品数据行
    - logger: 日志记录函数
    - reasoning_effort: 推理努力程度，用于控制成本。可选值: "low", "medium", "high"，默认为 "low"
    """
    product_name = row.get('modified_name', '')
    uniq_id = row.get('uniq_id', 'unknown_id')
    
    logger(f"[{uniq_id}] 开始处理基础产品优化: {product_name}")
    
    # 解析产品数据
    hard_design = str(row.get('hard_design', '') or '')
    core_features = str(row.get('core_features', '') or '')
    value_proposition = str(row.get('value_proposition', '') or '')
    
    # 检查产品文档是否为空
    if not hard_design and not core_features and not value_proposition:
        logger(f"[{uniq_id}] 警告: 产品文档为空，无法进行优化")
        return None
    
    # 构建提示词
    system_prompt = """你是一名资深产品经理，负责对产品进行优化改进，擅长深度挖掘用户痛点并设计针对性解决方案。你需要基于产品文档进行优化，确保优化后的设计在技术上可行、生产上可实现、成本上合理。最重要的是，所有优化必须直击用户痛点，解决用户在使用产品时遇到的实际问题。"""
    
    user_prompt = f"""
<task>
你是一名资深产品经理，负责对产品进行优化改进，提高产品的市场竞争力和用户价值。

请按照以下步骤完成优化任务：
1. **深度分析用户痛点**：仔细阅读当前产品的完整设计文档，从用户使用场景出发，识别用户在使用该产品时可能遇到的核心痛点，包括但不限于：
   - 功能缺失或不完善导致的用户困扰
   - 设计缺陷影响用户体验
   - 材料或工艺问题影响产品耐用性或舒适度
   - 使用不便或维护困难
   - 安全性或可靠性不足
   - 性价比不合理
   
2. **针对性优化设计**：针对识别出的用户痛点，设计具体的优化方案，确保：
   - 每个优化点都直接对应一个明确的用户痛点
   - 优化方案能够有效解决或缓解该痛点
   - 优化后的改进对用户来说是可见、可感知的
   - 优先解决影响用户核心体验的关键痛点
   
3. **确保优化的技术可行性**：优化后的产品设计必须可生产且无矛盾
4. **确保优化的成本合理性**：优化不应导致成本大幅增加，应优先考虑成本效益高的改进方案
5. **生成优化后的产品设计文档**
6. **生成优化前后对比**：清晰展示每个优化点的前后对比，让用户能够直观看到改进
7. **主动论证当前优化的可行性**
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

<optimization_rules>
1. **痛点导向原则**：所有优化必须直击用户痛点，每个优化点都应该对应一个明确的用户问题。不要做无关痛痒的改进，要解决用户真正关心的问题。
2. **用户价值优先**：优化的目的是提高产品的市场竞争力和用户价值，优先解决影响用户核心体验的关键问题。
3. **可见性要求**：优化后的改进对用户来说应该是可见、可感知的，用户能够明显感受到产品变得更好用了。
4. **技术可行性要求**：必须确保优化后的产品设计可生产且无矛盾。所有优化必须考虑实际生产的技术限制。
5. **成本合理性要求**：优化不应导致成本增加，应优先考虑成本效益高的改进方案，确保优化后的产品成本不超过原产品成本。
6. **针对性改进**：优化应该是有针对性的改进，每个改动都应该有明确的理由和对应的用户痛点。
7. **保持产品核心定位**：优化不应改变产品的核心定位和主要功能方向，但可以在保持核心的基础上增强功能。
8. **文档正式性要求**：优化后的产品文档（hard_design、core_features、value_proposition）必须是一个正式版的文档，不能显示出任何修改和编辑的痕迹。具体要求：
   - 绝对禁止使用括号标注修改的地方（如"（已优化）"、"（改进）"等）
   - 绝对禁止注明"从...改为..."、"原为...现为..."等修改说明
   - 绝对禁止使用任何形式的修改标记、注释或说明
   - 输出的文档应该像直接写出来的正式文档一样，完全看不出是经过修改的版本
   - 所有优化内容应该自然地融入到文档中，以正式、完整、流畅的方式呈现
</optimization_rules>

<output_format>
输出文档必须包括以下部分，且每个部分用XML标签包裹。不要添加任何无关内容。

<hard_design>
优化后的硬性设计部分：包括技术指标、设计细节、材料规格、尺寸等具体设计元素。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只进行有意义的优化改进，保持设计的完整性和一致性。
</hard_design>

<core_features>
优化后的核心功能介绍：描述产品的主要功能、优势、如何使用以及解决的用户痛点。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只进行有意义的优化改进，保持功能的完整性和一致性。
</core_features>

<value_proposition>
优化后的价值定位部分：说明产品的市场定位、目标用户群、竞争优势和独特卖点。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
注意：只进行有意义的优化改进，保持定位的完整性和一致性。
</value_proposition>

<feasibility_analysis>
优化可行性论证：主动论证当前优化的可行性，包括但不限于：
1. 技术可行性：说明优化后的设计在技术上是否可行，是否存在技术难点或限制
2. 生产可行性：说明优化后的设计是否可生产，生产过程中是否存在困难
3. 一致性检查：说明优化后的设计是否存在内部矛盾，各部分是否协调一致
4. 成本合理性：简要说明优化后产品的生产成本是否按要求不超过原产品成本
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</feasibility_analysis>

<optimization_comparison>
优化前后对比：以表格或列表形式清晰展示每个优化点的前后对比，格式如下：

**优化点1：[优化点名称]**
- **用户痛点**：[该优化点针对的用户痛点是什么]
- **优化前**：[优化前的具体表现]
- **优化后**：[优化后的具体表现]
- **改进效果**：[用户能感受到的具体改进效果]

**优化点2：[优化点名称]**
- **用户痛点**：[该优化点针对的用户痛点是什么]
- **优化前**：[优化前的具体表现]
- **优化后**：[优化后的具体表现]
- **改进效果**：[用户能感受到的具体改进效果]

...（继续列出所有优化点）

格式要求：
- 使用markdown格式，不要用```markdown```或``````包括
- 每个优化点必须明确对应一个用户痛点
- 对比要具体、清晰，让用户能够直观看到改进
- 至少列出3-5个核心优化点
</optimization_comparison>

<optimization_summary>
优化总结：简要说明本次优化的主要改进点、优化原因和预期效果。
格式：一段markdown风格的文本，不要用```markdown```或``````包括，格式工整规范。
</optimization_summary>

注意：
1. 所有的输出内容必须使用中文
2. 所有改动必须合理且有意义，且必须直击用户痛点
3. 保持产品设计的完整性和一致性
4. optimization_comparison部分必须详细列出所有核心优化点的前后对比
5. 所有优化内容应该自然地融入到文档中，以正式、完整的方式呈现，就像这是产品的最终正式版文档
</output_format>
"""
    
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    
    reasoning_text, result_content = await call_reasoner(system_prompt, user_prompt, reasoning_effort=reasoning_effort, debug=False)
    logger(f"[{uniq_id}] LLM思考过程: \n{reasoning_text}")
    logger(f"[{uniq_id}] LLM输出原文: \n{result_content}")
    
    # 解析LLM输出
    hard_design_new = ""
    core_features_new = ""
    value_proposition_new = ""
    feasibility_analysis = ""
    optimization_comparison = ""
    optimization_summary = ""
    
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
    
    optimization_comparison_match = re.search(r'<optimization_comparison>(.*?)</optimization_comparison>', result_content, re.DOTALL)
    if optimization_comparison_match:
        optimization_comparison = optimization_comparison_match.group(1).strip()
    else:
        raise ValueError("[优化前后对比]未正确生成")
    
    optimization_summary_match = re.search(r'<optimization_summary>(.*?)</optimization_summary>', result_content, re.DOTALL)
    if optimization_summary_match:
        optimization_summary = optimization_summary_match.group(1).strip()
    else:
        # 优化总结不是必需的，如果缺失则设为空字符串
        optimization_summary = ""
    
    logger(f"[{uniq_id}] 基础产品优化完成")
    
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
    new_row['optimization_comparison'] = optimization_comparison
    new_row['optimization_summary'] = optimization_summary
    
    return new_row


def filter_existing_base_optimization(product_row: pd.Series, existing_output_df: pd.DataFrame) -> bool:
    """
    判断某个产品是否需要基础优化。
    如果该产品的基础优化结果已存在于输出df中，则不需要处理。
    
    参数:
    - product_row: 产品行数据（Series），包含uniq_id（产品ID）
    - existing_output_df: 已存在的输出DataFrame
    
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
    
    # 检查输出文件中是否存在optimization_summary字段（作为基础优化的标识）
    if 'optimization_summary' not in existing_output_df.columns:
        return True
    
    # 检查输出文件中是否有该产品的行，且包含optimization_summary
    matching_rows = existing_output_df[existing_output_df['uniq_id'] == product_uniq_id]
    
    if matching_rows.empty:
        # 输出文件中没有该产品的行，需要处理
        return True
    
    # 检查是否有optimization_summary且不为空
    for _, row in matching_rows.iterrows():
        optimization_summary = row.get('optimization_summary')
        if pd.notna(optimization_summary) and str(optimization_summary).strip():
            # 找到了该产品的行且包含optimization_summary，说明已处理
            return False
    
    # 虽然有该产品的行，但没有optimization_summary，需要处理
    return True

