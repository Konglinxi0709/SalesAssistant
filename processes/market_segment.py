import json
import pandas as pd
from typing import Dict, Any, Callable
from utils.call_cuc_llm import call_reasoner

async def market_segment_task(row: Dict[str, Any], logger: Callable[[str], None]) -> Dict[str, Any]:
    """市场细分任务回调函数。"""
    uniq_id = row.get('uniq_id', 'unknown_id')
    product_name = row.get('modified_name', '')
    category = row.get('category', '')
    
    logger(f"[{uniq_id}] 开始市场细分分析: {product_name}")
    
    # 组装产品设计文档
    hard_design = row.get("hard_design", "")
    cost_estimate = row.get("cost_estimate", "")
    core_features = row.get("core_features", "")
    value_proposition = row.get("value_proposition", "")
    expenditure_category = row.get("expenditure_category", "")
    product_document = (
        "="*100 + "\n"
        + f"【产品名称】:{product_name}\n"
        + f"【产品类别】:{category}\n"
        + f"【产品所属的居民消费支出大类】:{expenditure_category}\n"
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
    
    # 系统提示词
    system_prompt = """你是一个市场分析专家，专门负责基于享乐价格法理论进行产品市场细分分析。你需要深入理解产品的价值属性，并据此划分用户群体。"""
    
    # 用户提示词
    user_prompt = f"""
<task>
基于享乐价格法（HedonicPriceIndexes）理论，对该产品进行市场细分分析。
</task>

<analysis_steps>
1. 依据产品价值属性定义原则，找出该产品的N个独立正交的价值属性，N的大小任意，尽可能的找全，并按对价值影响的大小从大到小排序
2. 根据当前产品的价值定位，依据潜在用户群体定义原则，定义出所有对该产品有需求的潜在用户群体。
3. 依据用户维度定义原则，找到M个与一个或多个价值属性对应，M≤5，优先找到对用户价值判断影响最显著的属性对应的用户维度，并且必须确保用户维度相互之间严格正交独立。
4. 依据维度划分原则，为每个用户维度分别给出3个部分的判定依据
</analysis_steps>

<principles>
<value_attribute_principles>
依据享乐价格法（HedonicPriceIndexes）理论，用户对一个产品的价值感知是由用户对该产品的各个特征属性的价值叠加得到，也就是说产品的价值可以拆解为各个属性的价值。  
依据此理论，可以为每个产品定义其多个价值属性，约束条件如下：  
1. 用户对各个价值属性的感知彼此之间应尽可能无关，即不存在用户认为属性A的价值越大则必然用户认为属性B的价值越大或越小这种情况  
2. 应尽可能不遗漏任何一个会独立影响产品整体价值的属性  
3. 不应该包含几乎不影响用户对该产品价值感知的属性
</value_attribute_principles>

<user_group_principles>
将市场上所有对该产品有需求的用户称作潜在用户群体（只要一个用户有可能购买该产品，他就应该被认为是用户整体的一部分），需要给出对任意用户是否属于潜在用户群体的准确判定标准。准确判定标准的撰写应该满足以下约束：  
1. 只根据撰写的判定标准，就能对总体市场上的每个用户准确判断出该用户是否属于潜在用户群体  
2. 潜在用户群体的判定标准应该基于用户自身的硬性客观条件。判定标准不应太过复杂，大致划分即可，判定标准划定的人群范围允许略大于实际有购买需求的真实潜在群体，但不能漏掉任何有可能购买的人群。
3. 你需要思考世界上的各类人群，仔细分析、深入思考他们是否存在对该产品的需求。
</user_group_principles>

<user_dimension_principles>
对于对该产品有需求的潜在用户群体整体，可以按某个特征或属性的个体情况对该整体进行划分，则此时依据的特征或属性称之为一个维度。如果在每个维度上将整体划分成$k$个部分，则设定$M$个维度就可以将整体划分为$k^M$个部分，称之为用户群体。  
为了尽可能全面、高效的分析市场上所有对某产品持不同感知的用户群体，需要让用户维度与该产品的价值属性一对一或一对多的对应，即每个维度上的用户差异影响且仅影响用户对对应的几个价值属性的感知。故依据$N$个价值属性，应该可以找出对应的$M$个用户维度，满足约束条件如下：  
1. $M≤min(5, N)$，即用户维度数既不超过价值属性个数，也不超过5
2. 每个用户维度有$l$个价值属性与之对应，$1≤l≤N$，每个价值属性最多对应一个用户维度。由于优先确保用户维度的正交独立以及个数不超过5，允许存在不对应任何用户维度的价值属性。
3. 每个用户维度影响且仅影响与之对应的价值属性，对其它价值属性几乎无影响  
4. 用户维度之间严格独立正交，即不存在当用户在A维度取值为a时，在B维度取值为b的概率就会大大增加或减少的情况  
</user_dimension_principles>

<dimension_division_principles>
在每个用户维度上，将潜在用户群体整体划分为三个部分，分别对应对当前产品在该价值属性上的表现感知为负价值（当前产品在该属性上的表现让用户对该产品的购买意愿和心理价格显著降低）、无价值（当前产品在该属性上的表现对用户对该产品的购买意愿和心理价格几乎无影响）和正价值（当前产品在该属性上的表现让用户对该产品的购买意愿和心理价格显著提升），给出三个部分的划分依据。这三个划分依据应该满足以下约束：  
1. 只根据撰写的单独一个部分的划分依据，就能对整体中的每个用户准确判断出该用户是否属于这个部分  
2. 在经过按第一个约束分类每个部分之后，**三个部分的并集应该不遗漏的等于整体**，且三个部分应该**无任何交集**  
3. 每个部分划分出的群体必须不能与潜在用户群体整体的判定标准存在任何矛盾
4. 你需要代入真实的用户，思考该产品能对这类用户带来的价值以及可能造成的负担和困扰有哪些，因为该产品在某个价值属性上的表现，什么样的用户会更喜欢这个产品，而什么样的用户会更不喜欢这个产品。你需要确保如果对三个部分中的用户进行实际调研，应该真的因为该产品在该价值属性的表现造成购买意愿的降低、不变和提高。尤其注意不要因为当前产品在某个属性上做了一些优化，就直接认为该属性对用户价值感知的影响是正向的，要从实际用户使用体验出发。
5. 尤其注意负价值群体必须因为该产品在该价值属性上的表现而降低购买意愿和心理价格，并且降低幅度应显著大于无价值群体。
6. 每个依据只能写用户自身的生活方式、性格特点、身体状态等客观因素，不能直接写明用户怎么看待该产品。你必须深度思考可能导致三个群体对该产品属性不同价值感知的深层原因，应分析到上述本质因素。
7. 当潜在用户群体整体中不可能出现对某个价值属性持某种感知的群体，你可以将该部分设置为空集，将对应的判定依据写为空字符串""即可，而不用强行编造一个并不合理的判定依据。这尤其可能出现在负价值部分。但是每个用户维度**最多只能有一个部分为空集**。
8. 每个部分判定依据的撰写上，直接写对该部分人的描述即可，例如“符合...条件的人”或“用户满足...条件”。禁止在判定依据中出现直接指明该部分属于负价值、无价值、无价值中哪一类的直接描述，包括直接出现这三个名称，或出现“中性”、“意愿低”等描述该部分分类的词汇。
</dimension_division_principles>
</principles>


<product_info>
{product_document}
</product_info>

<output_format>
输出必须是JSON格式，包含以下结构：
{{
    "user_group": "对该产品有需求的潜在用户群体整体的判定标准的详细描述",
    "total_attributes": [
        {{
            "attribute_name": "价值属性名称",
            "attribute_description": "该价值属性在当前产品上的具体体现描述"
        }}
    ],
    "segmentations": [
        {{
            "dimension_name": "用户维度名称",
            "attribute_names": ["对应的价值属性名称1", "对应的价值属性名称2", ...],
            "negative_value_criteria": "负价值用户群体判定依据的内容",
            "neutral_value_criteria": "无价值用户群体判定依据的内容", 
            "positive_value_criteria": "正价值用户群体判定依据的内容"
        }}
    ]
}}
不要使用```json```或``````包裹
</output_format>
"""
    
    logger(f"[{uniq_id}] 用户提示词：\n{user_prompt}")
    
    # 调用大模型
    resoning_text, output_text = await call_reasoner(system_prompt, user_prompt, reasoning_effort="high", debug=False)
    logger(f"[{uniq_id}] LLM思考过程: \n{resoning_text}")
    logger(f"[{uniq_id}] LLM输出原文: \n{output_text}")
    
    # 解析JSON输出
    try:
        output_json = json.loads(output_text)
        
        # 1. 检查顶层所有字段存在
        if "user_group" not in output_json:
            raise ValueError("JSON缺少必要字段: user_group")
        if "total_attributes" not in output_json:
            raise ValueError("JSON缺少必要字段: total_attributes")
        if "segmentations" not in output_json:
            raise ValueError("JSON缺少必要字段: segmentations")
        
        total_attributes = output_json["total_attributes"]
        segmentations = output_json["segmentations"]

        # 2. 验证 total_attributes 和 segmentations 都是非空数组
        if not isinstance(total_attributes, list) or len(total_attributes) == 0:
            raise ValueError("total_attributes 必须是长度大于0的数组")
        if not isinstance(segmentations, list) or len(segmentations) == 0:
            raise ValueError("segmentations 必须是长度大于0的数组")

        dimensions_count = len(segmentations)
        if dimensions_count > 5:
            raise ValueError(f"维度数量超过5，维度数量: {dimensions_count}")

        # 3. 验证每个 total_attribute 有必需字段，收集属性名
        attr_name_list = []
        for i, attr in enumerate(total_attributes):
            if not isinstance(attr, dict):
                raise ValueError(f"total_attributes[{i}] 不是字典类型")
            if "attribute_name" not in attr:
                raise ValueError(f"total_attributes[{i}] 缺少字段: attribute_name")
            if "attribute_description" not in attr:
                raise ValueError(f"total_attributes[{i}] 缺少字段: attribute_description")
            attr_name_list.append(attr["attribute_name"])

        # 4. 验证每个 segmentation 维度结构与字段
        # 收集所有已分配的属性名，验证唯一且全覆盖
        attr2segment = {}
        for i, segment in enumerate(segmentations):
            required_fields = [
                "attribute_names",
                "dimension_name",
                "negative_value_criteria",
                "neutral_value_criteria",
                "positive_value_criteria"
            ]
            if not isinstance(segment, dict):
                raise ValueError(f"segmentations[{i}] 不是字典类型")
            for field in required_fields:
                if field not in segment:
                    raise ValueError(f"segmentations[{i}] 缺少字段: {field}")
            attr_names = segment["attribute_names"]
            if not isinstance(attr_names, list) or len(attr_names) == 0:
                raise ValueError(f"segmentations[{i}].attribute_names 必须是非空数组")
            for attr_name in attr_names:
                if attr_name in attr2segment:
                    raise ValueError(f"属性 '{attr_name}' 在多个segmentations中重复分配")
                attr2segment[attr_name] = i  # 记录此属性分配的维度index

            # 新增：检查每个维度最多只能有一个部分为空集
            empty_criteria_count = 0
            criteria_fields = [
                segment.get("negative_value_criteria", ""),
                segment.get("neutral_value_criteria", ""),
                segment.get("positive_value_criteria", "")
            ]
            for criteria in criteria_fields:
                # 只计数严格为""的（空字符串）
                if isinstance(criteria, str) and criteria.strip() == "":
                    empty_criteria_count += 1
            if empty_criteria_count > 1:
                raise ValueError(f"segmentations[{i}] 至多只能有一个群体（负/无/正价值群体）为空集（判定依据为空字符串）。")

        # 5. 检查所有total_attributes中的属性恰好各对应一个维度
        # for attr_name in attr_name_list:
        #     if attr_name not in attr2segment:
        #         raise ValueError(f"价值属性 '{attr_name}' 未在任何 segmentation 的 attribute_names 中分配")
        # if len(attr2segment) != len(attr_name_list):
        #     extra = set(attr2segment.keys()) - set(attr_name_list)
        #     if extra:
        #         raise ValueError(f"存在total_attributes中未定义的属性被分配到segmentation: {extra}")

        logger(f"[{uniq_id}] 市场细分分析成功: {dimensions_count}个维度")

    except json.JSONDecodeError as e:
        logger(f"[{uniq_id}] JSON解析失败: {str(e)}")
        raise ValueError(f"[市场细分]JSON解析失败: {str(e)}")
    except Exception as e:
        logger(f"[{uniq_id}] 市场细分分析失败: {str(e)}")
        raise ValueError(f"[市场细分]分析失败: {str(e)}")
    
    # 更新行数据，添加新字段
    output_row = row.copy()
    output_row['dimensions_count'] = dimensions_count
    output_row['segmentation_result'] = json.dumps(output_json, ensure_ascii=False)
    
    logger(f"[{uniq_id}] **市场细分完成: {dimensions_count}个维度**")
    logger(f"[{uniq_id}] 细分结果: {json.dumps(output_json, ensure_ascii=False, indent=2)}")
    
    return output_row

def filter_existing_segmentation(row: Dict[str, Any], existing_output_df: pd.DataFrame) -> bool:
    """判断某一行是否需要市场细分分析。如果uniq_id已存在于输出df中且已有segmentation_result字段，则不需要分析。"""
    uniq_id = row.get('uniq_id')
    if uniq_id is None:
        # 如果没有uniq_id，默认处理它
        return True 
    
    # 如果输出DataFrame为空，则所有行都需要处理
    if existing_output_df.empty:
        return True

    # 检查uniq_id是否在现有输出文件中，且是否已有segmentation_result字段
    if uniq_id in existing_output_df['uniq_id'].values:
        # 找到对应的行
        matching_rows = existing_output_df[existing_output_df['uniq_id'] == uniq_id]
        if not matching_rows.empty:
            # 检查是否已有segmentation_result字段且不为空
            if 'segmentation_result' in matching_rows.columns:
                segmentation_value = matching_rows.iloc[0]['segmentation_result']
                if pd.notna(segmentation_value) and str(segmentation_value).strip() != '':
                    return False  # 已有细分结果，不需要重新分析
    
    return True  # 需要分析
