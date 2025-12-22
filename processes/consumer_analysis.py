from typing import Any
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_dimension_values(uniq_id: str, product_uniq_id: str) -> list:
    """从uniq_id中解析维度取值，支持任意维度数量"""
    # 移除product_uniq_id部分，剩下的就是维度编码
    dimension_part = uniq_id.replace(product_uniq_id + '_', '')
    dimension_codes = dimension_part.split('_')
    
    # 将0,1,2映射为-1,0,1
    return [int(x) - 1 for x in dimension_codes]


def to_plus_minus_features(dim_values: list[int]) -> list[float]:
    """
    将每个维度取值 x_i ∈ {-1,0,1} 变换为 (x_i^+, x_i^-)，用于双参数线性拟合：

    x_i^+ = 1 (当 x_i=1) 否则 0
    x_i^- = -1 (当 x_i=-1) 否则 0

    则 v = β0 + Σ(β_i^+ x_i^+ + β_i^- x_i^-) + ε

    返回特征顺序为：[x_1^+, x_1^-, x_2^+, x_2^-, ...]
    """
    feats: list[float] = []
    for x in dim_values:
        x_plus = 1.0 if x == 1 else 0.0
        x_minus = -1.0 if x == -1 else 0.0
        feats.extend([x_plus, x_minus])
    return feats


def calculate_dimension_variance(proportions: list) -> float:
    """计算维度的方差"""
    if len(proportions) != 3:
        return 0.0
    
    # 计算期望 E[x]
    expectation = (-1) * proportions[0] + 0 * proportions[1] + 1 * proportions[2]
    
    # 计算E[x^2]
    expectation_sq = (1 * proportions[0]) + (0 * proportions[1]) + (1 * proportions[2])
    
    # 方差 = E[x^2] - (E[x])^2
    variance = expectation_sq - expectation**2
    return variance


def analyze_product_market(product_data: pd.DataFrame) -> dict:
    """
    分析单个产品的市场表现，计算各项指标
    
    参数:
    - product_data: 包含同一产品所有消费者数据的DataFrame
    
    返回:
    - 包含各项指标的字典，如果分析失败返回None
    """
    if len(product_data) == 0:
        return None
    
    # 提取维度比例信息
    try:
        proportion_info = json.loads(product_data.iloc[0]['proportion_estimate'])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"警告: 解析proportion_estimate时出错: {e}")
        return None

    # 提取分群结果，建立属性到维度的映射（含正向/反向）
    attr_dim_map = {}
    try:
        segmentation_result_str = product_data.iloc[0].get('segmentation_result', '{}')
        segmentation_result = json.loads(segmentation_result_str) if segmentation_result_str else {}
        for dim_idx, seg in enumerate(segmentation_result.get('segmentations', [])):
            for attr in seg.get('positive_attribute_names', []):
                attr_dim_map[attr] = (dim_idx, "正向")
            for attr in seg.get('negative_attribute_names', []):
                attr_dim_map[attr] = (dim_idx, "反向")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"警告: 解析segmentation_result时出错: {e}")
        attr_dim_map = {}
    
    num_dimensions = len(proportion_info)
    
    # 准备线性回归数据
    X = []
    y = []
    group_proportions = []
    group_uniq_ids = []
    # 收集属性打分与维度取值，用于计算k值
    attr_points = {}  # attr_name -> list of (dim_value, score, direction)
    product_uniq_id = product_data.iloc[0]['product_uniq_id']
    
    for _, row in product_data.iterrows():
        try:
            # 解析维度值
            dim_values = parse_dimension_values(row['uniq_id'], product_uniq_id)
            
            # 确保维度数量匹配
            if len(dim_values) != num_dimensions:
                print(f"警告: 维度数量不匹配，期望{num_dimensions}，实际{len(dim_values)}，跳过该行")
                continue
                
            X.append(to_plus_minus_features(dim_values))
            group_uniq_ids.append(str(row['uniq_id']))
            
            # 获取心理价格作为因变量
            psychological_price_str = str(row['psychological_price']).replace('￥', '').strip()
            psychological_price = float(psychological_price_str)
            y.append(psychological_price)

            # 收集属性打分数据
            try:
                attr_analysis = json.loads(row.get('attribute_analysis', '[]'))
            except (json.JSONDecodeError, TypeError):
                attr_analysis = []
            for item in attr_analysis:
                attr_name = item.get('attribute_name')
                if attr_name in attr_dim_map:
                    dim_idx, direction = attr_dim_map[attr_name]
                    if dim_idx < len(dim_values):
                        dim_val = dim_values[dim_idx]  # -1,0,1
                        score_val = item.get('attribute_score')
                        if isinstance(score_val, (int, float)):
                            attr_points.setdefault(attr_name, []).append(
                                (dim_val, float(score_val), direction)
                            )
            
            # 计算该用户群体的比例
            group_prop = 1.0
            for i, dim_val in enumerate(dim_values):
                dim_props = proportion_info[i]['proportions']
                # 将-1,0,1映射到0,1,2索引
                prop_index = dim_val + 1
                if 0 <= prop_index < len(dim_props):
                    group_prop *= dim_props[prop_index]
                else:
                    group_prop = 0.0
                    break
            
            group_proportions.append(group_prop)
            
        except (ValueError, IndexError, KeyError) as e:
            print(f"解析错误: {e}, 跳过该行")
            continue
    
    if len(X) == 0:
        print(f"错误: 没有有效数据用于产品 {product_data.iloc[0].get('modified_name', 'unknown')}")
        return None
    
    X = np.array(X)
    y = np.array(y)
    group_proportions = np.array(group_proportions)
    group_uniq_ids = np.array(group_uniq_ids, dtype=object)
    
    # 线性回归拟合
    model = LinearRegression()
    model.fit(X, y)
    
    # 预测值用于计算误差
    y_pred = model.predict(X)
    
    # 计算拟合误差
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # 获取系数（双参数：每个维度对应 β_i^+ 与 β_i^-）
    beta_0 = model.intercept_  # 截距项β0
    coefs = np.array(model.coef_, dtype=float)  # 长度为 2M
    if coefs.shape[0] != 2 * num_dimensions:
        raise ValueError(f"系数维度异常：期望{2 * num_dimensions}，实际{coefs.shape[0]}")
    beta_plus = coefs[0::2]   # β_i^+（对应x_i^+）
    beta_minus = coefs[1::2]  # β_i^-（对应x_i^-）
    
    # 计算各维度的重要性指标
    importance_indicators = []
    dimension_variances = []
    
    for i in range(num_dimensions):
        dim_variance = calculate_dimension_variance(proportion_info[i]['proportions'])
        dimension_variances.append(dim_variance)
        # 双参数下的“重要性”采用合成强度：sqrt(β+^2 + β-^2) * sqrt(Var)
        beta_strength = float(np.sqrt(beta_plus[i] ** 2 + beta_minus[i] ** 2))
        importance = beta_strength * np.sqrt(dim_variance) if dim_variance > 0 else 0.0
        importance_indicators.append(importance)
    
    # 计算最优定价和最大总利润
    try:
        cost_estimate_str = str(product_data.iloc[0]['cost_estimate']).replace('￥', '').strip()
        # cost = float(cost_estimate_str)
        cost = 0.0 # 放弃使用成本，直接计算最大销售额
    except (ValueError, KeyError) as e:
        print(f"警告: 无法解析成本 ({e})，跳过最优定价计算")
        cost = 0.0
    
    candidate_prices = sorted(set(y))  # 所有不重复的用户心理价格
    max_profit = 0.0
    optimal_price = cost if cost > 0 else (candidate_prices[0] if candidate_prices else 0.0)
    
    for price in candidate_prices:
        if price < cost:  # 跳过低于成本的定价
            continue
            
        total_profit = 0.0
        for i, valuation in enumerate(y):
            if valuation >= price:
                profit_per_customer = (price - cost) * group_proportions[i]
                total_profit += profit_per_customer
        
        if total_profit > max_profit:
            max_profit = total_profit
            optimal_price = price
        # 如果利润相同，选择更高的价格（提高利润率）
        elif abs(total_profit - max_profit) < 1e-9 and price > optimal_price:
            optimal_price = price

    # 中心群体：估值恰好等于最优定价的所有群体
    center_group_ids: list[str] = []
    try:
        # 所有群体按估值从小到大排序（稳定排序，保证同估值时相对顺序可复现）
        order = np.argsort(y, kind="stable")
        sorted_y = y[order]
        sorted_ids = group_uniq_ids[order]

        # 中心群体：估值恰好等于最优定价的所有群体
        center_positions = np.where(sorted_y == optimal_price)[0].tolist()
        center_group_ids = [str(sorted_ids[p]) for p in center_positions]
    except Exception as e:
        print(f"警告: 中心群体计算失败: {e}")
        center_group_ids = []
    
    # 基于属性打分与维度取值计算k值（双参数：k^+ 与 k^-；反向属性取相反数）
    # 注意：这里仍保留LinearRegression默认截距，以适配“无价值群体平均分不一定为0”的情况。
    k_attr_scores = {}
    for attr_name, records in attr_points.items():
        if not records:
            continue
        xs = [r[0] for r in records]  # -1,0,1
        ys = [r[1] for r in records]
        direction = records[0][2]
        k_plus = 0.0
        k_minus = 0.0
        if len(set(xs)) >= 2:
            try:
                # 构造(x^+, x^-)特征：x^+=1当x=1；x^-=-1当x=-1
                X_attr = np.array([[1.0 if x == 1 else 0.0, -1.0 if x == -1 else 0.0] for x in xs], dtype=float)
                lm_attr = LinearRegression()
                lm_attr.fit(X_attr, np.array(ys, dtype=float))
                coef_attr = np.array(lm_attr.coef_, dtype=float)
                if coef_attr.shape[0] == 2:
                    k_plus = float(coef_attr[0])
                    k_minus = float(coef_attr[1])
            except Exception:
                k_plus = 0.0
                k_minus = 0.0

        # 若该属性属于“反向”，则将k整体取相反数，使其含义仍为“属性更好 -> 分数更高”
        if direction == "反向":
            k_plus = -k_plus
            k_minus = -k_minus

        k_attr_scores[attr_name] = {"k_plus": k_plus, "k_minus": k_minus}

    return {
        'beta_0': float(beta_0),
        'beta_plus_coefficients': json.dumps([float(x) for x in beta_plus], ensure_ascii=False),
        'beta_minus_coefficients': json.dumps([float(x) for x in beta_minus], ensure_ascii=False),
        'dimension_variances': json.dumps([float(x) for x in dimension_variances], ensure_ascii=False),
        'importance_indicators': json.dumps([float(x) for x in importance_indicators], ensure_ascii=False),
        'optimal_price': float(optimal_price),
        'max_total_profit': float(max_profit),
        'optimal_price_center_group_ids': json.dumps(center_group_ids, ensure_ascii=False),
        'r_squared': float(model.score(X, y)),
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'k_attr_scores': json.dumps(k_attr_scores, ensure_ascii=False),
    }


def run_consumer_analysis(input_file: str, output_file: str):
    """
    将consumer_feedback_data.csv处理为consumer_analysis.csv，添加市场分析指标
    
    参数:
    - input_file: 输入的CSV文件路径（consumer_feedback_data.csv）
    - output_file: 输出的CSV文件路径（consumer_analysis.csv）
    """
    # 读取输入文件
    df = pd.read_csv(input_file)
    print(f"成功读取数据，共 {len(df)} 行记录")
    
    # 检查必要的列是否存在
    required_columns = ['product_uniq_id', 'uniq_id', 'proportion_estimate', 'psychological_price', 'cost_estimate']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少必要的列: {missing_columns}")
    
    # 按产品分组
    product_groups = df.groupby('product_uniq_id')
    print(f"发现 {len(product_groups)} 个不同产品")
    
    # 存储处理后的数据
    processed_rows = []
    analyzed_count = 0
    failed_count = 0
    
    # 遍历每个产品
    for product_id, product_data in product_groups:
        product_name = product_data.iloc[0].get('modified_name', product_id)
        print(f"正在分析产品: {product_name} (共 {len(product_data)} 个消费者)")
        
        # 分析该产品
        analysis_result = analyze_product_market(product_data)
        
        if analysis_result:
            # 为该产品的所有消费者行添加分析指标
            for _, row in product_data.iterrows():
                new_row = row.copy()
                # 添加所有分析指标
                for key, value in analysis_result.items():
                    new_row[key] = value
                processed_rows.append(new_row)
            analyzed_count += 1
        else:
            # 即使分析失败，也保留原始数据，但添加空指标字段（设为NaN）以保持输出结构一致
            print(f"警告: 产品 {product_name} 分析失败，保留原始数据但指标字段为空")
            indicator_fields = [
                'beta_0', 'beta_plus_coefficients', 'beta_minus_coefficients', 'dimension_variances',
                'importance_indicators', 'optimal_price', 'max_total_profit',
                'optimal_price_center_group_ids',
                'r_squared', 'mse', 'mae', 'rmse', 'k_attr_scores'
            ]
            for _, row in product_data.iterrows():
                new_row = row.copy()
                # 添加空指标字段
                for field in indicator_fields:
                    new_row[field] = None
                processed_rows.append(new_row)
            failed_count += 1
    
    if not processed_rows:
        print("警告: 没有生成任何数据")
        return
    
    # 创建DataFrame
    result_df = pd.DataFrame(processed_rows)
    
    # 保存到输出文件
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n处理完成!")
    print(f"  成功分析产品数: {analyzed_count}")
    print(f"  分析失败产品数: {failed_count}")
    print(f"  总消费者行数: {len(result_df)}")
    print(f"  结果已保存到: {output_file}")

