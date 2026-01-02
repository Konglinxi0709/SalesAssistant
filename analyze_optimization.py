import pandas as pd
import json
import re
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.stats import kendalltau
import difflib
import argparse

# 设置matplotlib后端，防止在无GUI环境下报错
matplotlib.use('Agg')

# 设置特定中文字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

def safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except:
            return None
    return x

def parse_price(price_str):
    """
    解析价格字符串，移除货币符号并转换为浮点数
    例如: "485.60￥" -> 485.60
    """
    if isinstance(price_str, (int, float)):
        val = float(price_str)
        return val if val > 0 else 1.0
    if not isinstance(price_str, str):
        return 1.0
    try:
        clean_str = re.sub(r'[^\d.]', '', price_str)
        val = float(clean_str)
        return val if val > 0 else 1.0
    except:
        return 1.0

def _parse_group_dim_values_from_uniq_id(uniq_id: str, num_dims: int) -> list:
    """从uniq_id末尾解析维度取值(0/1/2)。"""
    parts = str(uniq_id).split("_")
    if len(parts) < num_dims:
        return None
    try:
        return [int(x) for x in parts[-num_dims:]]
    except Exception:
        return None

def _calc_group_proportion(proportion_estimate: list, dim_values_012: list[int]) -> float:
    """
    计算群体占比 N(x)=∏ φ_i(x_i)，其中dim_values_012为0/1/2（对应负/无/正）。
    """
    if not isinstance(proportion_estimate, list):
        return 0.0
    prop = 1.0
    for i, v in enumerate(dim_values_012):
        try:
            probs = proportion_estimate[i]["proportions"]
            prop *= float(probs[v])
        except Exception:
            return 0.0
    return float(prop)

def generate_optimization_table_for_analysis(product_row: pd.Series) -> list:
    """
    为分析目的重新生成优化优先顺序表（简化版，只返回必要信息）
    
    参数:
    - product_row: 产品行数据
    
    返回:
    - 优化选项列表，每个选项包含 attr_name, direction, reference_gid
    """
    from processes.product_optimization import generate_optimization_table
    
    try:
        optimization_table = generate_optimization_table(product_row)
        return optimization_table
    except Exception as e:
        print(f"警告: 生成优化优先顺序表失败: {e}")
        return []

def count_adopted_suggestions_for_group(group_id: str, suggestion_analysis: list, optimization_table: list) -> int:
    """
    计算某个群体被采纳的建议数量
    
    参数:
    - group_id: 群体ID
    - suggestion_analysis: 建议分析列表，每个元素包含 attr_name, direction, adopted
    - optimization_table: 优化优先顺序表，每个元素包含 attr_name, direction, reference_consumers
    
    返回:
    - 被采纳的建议数量（该群体出现在reference_consumer_list中并且对应的优化方向被采纳的次数）
    """
    if not suggestion_analysis or not isinstance(suggestion_analysis, list):
        return 0
    
    if not optimization_table or not isinstance(optimization_table, list):
        return 0
    
    # 构建建议分析的映射：attr_name + direction -> adopted
    suggestion_map = {}
    for suggestion in suggestion_analysis:
        if not isinstance(suggestion, dict):
            continue
        attr_name = suggestion.get('attr_name', '')
        direction = suggestion.get('direction', '')
        adopted = suggestion.get('adopted', False)
        if attr_name and direction:
            key = f"{attr_name} - {direction}"
            suggestion_map[key] = adopted
    
    # 统计被采纳的建议数量
    count = 0
    group_id_str = str(group_id).strip()
    
    for opt in optimization_table:
        if not isinstance(opt, dict):
            continue
        
        attr_name = opt.get('attr_name', '')
        direction = opt.get('direction', '')
        reference_consumers = opt.get('reference_consumers', [])
        
        if not attr_name or not direction:
            continue
        
        # 检查该群体是否在reference_consumer_list中
        in_reference_list = False
        for consumer in reference_consumers:
            if not isinstance(consumer, dict):
                continue
            consumer_gid = str(consumer.get('gid', '')).strip()
            # 排除无效的gid（如"（无）"、"（空）"）
            if consumer_gid and consumer_gid not in ["（无）", "（空）"] and consumer_gid == group_id_str:
                in_reference_list = True
                break
        
        if not in_reference_list:
            continue
        
        # 检查该优化方向是否被采纳
        key = f"{attr_name} - {direction}"
        if key in suggestion_map and suggestion_map[key]:
            count += 1
    
    return count

def plot_comparison_histogram(groups_old, groups_new, title, filename, optimal_price_old: float, optimal_price_new: float, sort_by_old: bool = True, suggestion_analysis=None, optimization_table=None, num_dims=0):
    """
    绘制新老估值对比柱状图
    
    参数:
    - groups_old: list of dict { 'valuation': float, 'proportion': float, 'group_id': str }
    - groups_new: list of dict { 'valuation': float, 'proportion': float, 'group_id': str }
    - title: 图表标题
    - filename: 保存文件名
    - optimal_price_old: 优化前最优定价
    - optimal_price_new: 优化后最优定价
    - sort_by_old: True表示按老估值排序，False表示按新估值排序
    - suggestion_analysis: 建议分析列表（用于计算采纳建议数量）
    - optimization_table: 优化优先顺序表（用于计算采纳建议数量）
    - num_dims: 维度数量（用于生成dim_code标注）
    """
    if not groups_old or not groups_new:
        return
    
    # 创建group_id到数据的映射
    old_map = {g.get('group_id', ''): g for g in groups_old}
    new_map = {g.get('group_id', ''): g for g in groups_new}
    
    # 获取所有group_id的交集
    common_ids = set(old_map.keys()) & set(new_map.keys())
    if not common_ids:
        return
    
    # 根据sort_by_old决定排序方式
    if sort_by_old:
        # 按老估值排序
        sorted_groups = sorted(common_ids, key=lambda gid: old_map[gid].get('valuation', 0.0))
        sort_label = "Old Valuation"
    else:
        # 按新估值排序
        sorted_groups = sorted(common_ids, key=lambda gid: new_map[gid].get('valuation', 0.0))
        sort_label = "New Valuation"
    
    # 准备数据
    n = len(sorted_groups)
    base_width = min(0.01, 0.25 / max(1, n))
    
    # 计算宽度（基于老估值的占比）
    widths = np.array([max(0.0, float(old_map[gid].get("proportion", 0.0))) for gid in sorted_groups], dtype=float)
    total_w = float(widths.sum())
    if total_w <= 0:
        widths = np.ones_like(widths) / max(1, len(widths))
        total_w = float(widths.sum())
    else:
        widths = widths + base_width
        widths = widths / float(widths.sum())
    
    lefts = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
    
    # 获取老估值和新估值
    heights_old = np.array([float(old_map[gid].get("valuation", 0.0)) for gid in sorted_groups], dtype=float)
    heights_new = np.array([float(new_map[gid].get("valuation", 0.0)) for gid in sorted_groups], dtype=float)
    
    # 计算差值
    heights_diff = heights_new - heights_old
    
    # 确定颜色规则
    if sort_by_old:
        # 按老估值排序：下层柱颜色取决于新估值是否高于新定价线
        colors_old = ["#87CEEB" if h_new >= float(optimal_price_new) else "#bdbdbd" for h_new in heights_new]
        # 上层柱：负值红色，正值绿色
        colors_diff = ["#d73027" if h_diff < 0 else "#1a9850" for h_diff in heights_diff]
    else:
        # 按新估值排序：下层柱颜色取决于老估值是否高于老定价线
        colors_old = ["#87CEEB" if h_old >= float(optimal_price_old) else "#bdbdbd" for h_old in heights_old]
        # 上层柱：负值红色，正值绿色
        colors_diff = ["#d73027" if h_diff < 0 else "#1a9850" for h_diff in heights_diff]
    
    # 计算每个群体被采纳的建议数量
    adopted_counts = []
    if suggestion_analysis and optimization_table:
        for gid in sorted_groups:
            count = count_adopted_suggestions_for_group(gid, suggestion_analysis, optimization_table)
            adopted_counts.append(count)
    else:
        adopted_counts = [0] * n
    
    plt.figure(figsize=(14, 6))
    
    # 绘制下层柱（老估值）
    plt.bar(lefts, heights_old, width=widths, align="edge", color=colors_old, edgecolor="white", linewidth=0.2, label="Old Valuation")
    
    # 绘制上层柱（差值部分）
    for i, (left, w, h_old, h_new, h_diff, color) in enumerate(zip(lefts, widths, heights_old, heights_new, heights_diff, colors_diff)):
        if h_diff > 0:
            plt.bar(left, h_diff, width=w, bottom=h_old, align="edge", color=color, edgecolor="white", linewidth=0.2, alpha=0.7)
        elif h_diff < 0:
            # 如果新估值小于老估值，用红色显示减少的部分
            plt.bar(left, abs(h_diff), width=w, bottom=h_new, align="edge", color=color, edgecolor="white", linewidth=0.2, alpha=0.7)
    
    # 标注：维度取值和采纳建议数量
    # 使用柱子的最高处（max(老估值, 新估值)）作为标注基准
    max_h = float(np.max(np.maximum(heights_old, heights_new))) if len(heights_old) else 0.0
    y_offset_dim_code = max(1.0, 0.01 * max_h)  # 维度取值的偏移
    y_offset_count = y_offset_dim_code + max(4.0, 0.04 * max_h)  # 采纳建议数量的偏移（在维度取值上方，距离加倍）
    
    for i, (gid, left, w, h_old, h_new, count) in enumerate(zip(sorted_groups, lefts, widths, heights_old, heights_new, adopted_counts)):
        x = float(left) + float(w) / 2.0
        
        # 使用柱子的最高处作为标注基准
        max_height = max(h_old, h_new)
        
        # 1. 标注维度取值（dim_code）
        dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
        if dim_vals is not None:
            dim_code = "".join(str(int(v)) for v in dim_vals)
            y_dim_code = float(max_height) + y_offset_dim_code
            plt.text(x, y_dim_code, dim_code, rotation=45, ha="right", va="bottom", fontsize=6, color="black")
        
        # 2. 标注采纳建议数量（在维度取值上方，字体大两号，橙色）
        if count > 0:
            y_count = float(max_height) + y_offset_count
            plt.text(x, y_count, str(count), ha="center", va="bottom", fontsize=10, color="#fc8d59", weight="bold")
    
    # 添加最优定价线
    plt.axhline(y=float(optimal_price_old), color="red", linestyle="--", linewidth=1.2, label=f"Old Optimal Price ({optimal_price_old:.2f})")
    plt.axhline(y=float(optimal_price_new), color="blue", linestyle="--", linewidth=1.2, label=f"New Optimal Price ({optimal_price_new:.2f})")
    
    plt.title(title)
    plt.xlabel(f"Cumulative Group Proportion (Sorted by {sort_label})")
    plt.ylabel("Valuation (Psychological Price)")
    plt.xlim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def generate_diff(old_text: str, new_text: str) -> str:
    """
    生成unified diff格式的diff文本
    """
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, lineterm='', n=3)
    return ''.join(diff)

def analyze_optimization_data(input_file, output_file):
    """
    分析优化后的数据，生成优化分析报告
    
    参数:
    - input_file: 优化分析数据文件（optimization_analysis_data.csv）
    - output_file: 输出Markdown文件路径
    """
    # 确保图片输出目录存在
    # 从输出文件名中提取基础名称（不含扩展名），创建 基础名称_pictures 文件夹
    output_basename = os.path.splitext(os.path.basename(output_file))[0]
    img_dir_name = f"{output_basename}_pictures"
    img_dir = os.path.join(os.path.dirname(output_file), img_dir_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    print(f"成功读取数据，共 {len(df)} 个产品")
    
    # 存储所有产品的数据
    products_data = []
    
    # 全局统计
    total_sales_improved_count = 0
    total_sales_improvements = []
    kendall_tau_values = []
    # 用于统计不同建议采纳次数的群体的平均相对优化值
    adoption_count_to_improvements = {}  # {adoption_count: [improvement_ratios]}
    # 用于统计群体数量与相对优化值的关系
    group_count_vs_improvement = []  # [(group_count, improvement_ratio), ...]
    
    # 1. 遍历每一行数据（一行一个产品）
    for index, row in df.iterrows():
        product_name = row.get('modified_name', 'unknown')
        product_uniq_id = row.get('uniq_id', 'unknown')
        
        # 获取优化前后的分析指标
        max_total_profit_old = row.get('max_total_profit_org', 0.0)
        max_total_profit_new = row.get('max_total_profit', 0.0)
        optimal_price_old = row.get('optimal_price_org', 0.0)
        optimal_price_new = row.get('optimal_price', 0.0)
        retail_price = parse_price(row.get('retail_price', '0'))
        
        # 计算销售总额相对优化值
        improvement_ratio = None
        if max_total_profit_old > 0:
            improvement_ratio = (max_total_profit_new - max_total_profit_old) / max_total_profit_old * 100.0
            total_sales_improvements.append(improvement_ratio)
            if max_total_profit_new > max_total_profit_old:
                total_sales_improved_count += 1
        
        # 获取用户数据（优化后的）
        users_data = safe_json_loads(row.get('users_data', '[]'))
        if not isinstance(users_data, list):
            continue
        
        # 获取优化前的用户数据（从users_data_org字段）
        users_data_org = safe_json_loads(row.get('users_data_org', '[]'))
        if not isinstance(users_data_org, list):
            users_data_org = []
        
        # 构建优化前的用户ID到心理价格的映射
        original_valuations = {}
        for user_data_org in users_data_org:
            user_uniq_id = user_data_org.get('user_uniq_id', '') or user_data_org.get('uniq_id', '')
            if user_uniq_id:
                price_str = user_data_org.get('psychological_price', '0')
                original_valuations[user_uniq_id] = parse_price(price_str)
        
        # 获取proportion_estimate
        proportion_estimate = safe_json_loads(row.get('proportion_estimate'))
        if not isinstance(proportion_estimate, list):
            continue
        
        num_dims = len(proportion_estimate)
        
        # 计算每个群体的老估值和新估值
        groups_old = []
        groups_new = []
        unique_group_ids = set()  # 用于统计唯一群体数量
        
        for user_data in users_data:
            user_uniq_id = user_data.get('user_uniq_id', '')
            if not user_uniq_id:
                continue
            
            # 解析维度取值
            dim_values = _parse_group_dim_values_from_uniq_id(user_uniq_id, num_dims)
            if dim_values is None:
                continue
            
            # 统计唯一群体
            unique_group_ids.add(user_uniq_id)
            
            # 计算群体占比
            proportion = _calc_group_proportion(proportion_estimate, dim_values)
            
            # 获取心理价格（优化后的）
            price_str_new = user_data.get('psychological_price', '0')
            valuation_new = parse_price(price_str_new)
            
            # 获取优化前的心理价格（从users_data_org中获取）
            valuation_old = original_valuations.get(user_uniq_id, valuation_new)
            
            groups_old.append({
                'group_id': user_uniq_id,
                'valuation': valuation_old,
                'proportion': proportion
            })
            groups_new.append({
                'group_id': user_uniq_id,
                'valuation': valuation_new,
                'proportion': proportion
            })
        
        # 统计群体数量并收集数据
        group_count = len(unique_group_ids)
        if improvement_ratio is not None:
            group_count_vs_improvement.append((group_count, improvement_ratio))
        
        # 计算Kendall Tau相关系数
        if len(groups_old) > 1 and len(groups_new) > 1:
            # 按老估值排序获取排名
            sorted_old = sorted(groups_old, key=lambda g: g['valuation'])
            old_ranks = {g['group_id']: i for i, g in enumerate(sorted_old)}
            
            # 按新估值排序获取排名
            sorted_new = sorted(groups_new, key=lambda g: g['valuation'])
            new_ranks = {g['group_id']: i for i, g in enumerate(sorted_new)}
            
            # 获取共同的group_id
            common_ids = set(old_ranks.keys()) & set(new_ranks.keys())
            if len(common_ids) > 1:
                old_rank_list = [old_ranks[gid] for gid in common_ids]
                new_rank_list = [new_ranks[gid] for gid in common_ids]
                
                tau, p_value = kendalltau(old_rank_list, new_rank_list)
                if not np.isnan(tau):
                    kendall_tau_values.append(tau)
        
        # 获取suggestion_analysis
        suggestion_analysis = safe_json_loads(row.get('suggestion_analysis', '[]'))
        
        # 重新生成优化优先顺序表（用于确定参考群体）
        try:
            optimization_table = generate_optimization_table_for_analysis(row)
        except Exception as e:
            print(f"警告: 产品 {product_name} 生成优化优先顺序表失败: {e}")
            optimization_table = []
        
        # 获取属性映射信息（用于计算优化方向）
        segmentation = safe_json_loads(row.get('segmentation_result'))
        attr_dim_map = {}  # {attr_name: {'dim_idx': int, 'direction': '正向'/'反向'}}
        if segmentation:
            dims = segmentation.get('segmentations', [])
            for idx, dim in enumerate(dims):
                for attr_name in dim.get('positive_attribute_names', []):
                    attr_dim_map[attr_name] = {'dim_idx': idx, 'direction': '正向'}
                for attr_name in dim.get('negative_attribute_names', []):
                    attr_dim_map[attr_name] = {'dim_idx': idx, 'direction': '反向'}
        
        # 计算每个群体的采纳建议数量和相对优化值（用于全局统计）
        # 同时收集用于散点图的数据
        scatter_data = []  # [(adoption_ratio, valuation_improvement), ...]
        
        for user_data in users_data:
            user_uniq_id = user_data.get('user_uniq_id', '')
            if not user_uniq_id:
                continue
            
            # 解析维度取值
            dim_values = _parse_group_dim_values_from_uniq_id(user_uniq_id, num_dims)
            if dim_values is None:
                continue
            
            # 计算采纳建议数量
            adoption_count = 0
            if suggestion_analysis and optimization_table:
                adoption_count = count_adopted_suggestions_for_group(user_uniq_id, suggestion_analysis, optimization_table)
            
            # 计算相对优化值（(新估值-老估值)/老估值）
            valuation_old = original_valuations.get(user_uniq_id, None)
            valuation_new = parse_price(user_data.get('psychological_price', '0'))
            if valuation_old is not None and valuation_old > 0:
                relative_improvement = (valuation_new - valuation_old) / valuation_old * 100.0
                if adoption_count not in adoption_count_to_improvements:
                    adoption_count_to_improvements[adoption_count] = []
                adoption_count_to_improvements[adoption_count].append(relative_improvement)
            
            # 计算该群体对应的优化方向采纳占比
            if suggestion_analysis and optimization_table and attr_dim_map:
                # 计算该群体对应的所有优化方向
                total_directions = 0
                adopted_directions = 0
                
                for attr_name, attr_info in attr_dim_map.items():
                    dim_idx = attr_info['dim_idx']
                    direction_type = attr_info['direction']
                    
                    if dim_idx >= len(dim_values):
                        continue
                    
                    dim_val = dim_values[dim_idx]
                    
                    # 确定优化方向（考虑反向属性的交换）
                    if direction_type == '正向':
                        if dim_val == 0:
                            opt_direction = '对负价值群体减少负面影响'
                        elif dim_val == 2:
                            opt_direction = '对正价值群体提高满意度'
                        else:
                            continue  # 无价值群体不计算
                    else:  # 反向
                        if dim_val == 0:
                            opt_direction = '对正价值群体提高满意度'  # 交换
                        elif dim_val == 2:
                            opt_direction = '对负价值群体减少负面影响'  # 交换
                        else:
                            continue  # 无价值群体不计算
                    
                    total_directions += 1
                    
                    # 检查该优化方向是否被采纳
                    for analysis_item in suggestion_analysis:
                        if not isinstance(analysis_item, dict):
                            continue
                        if (analysis_item.get('attr_name') == attr_name and 
                            analysis_item.get('direction') == opt_direction and 
                            analysis_item.get('adopted', False)):
                            adopted_directions += 1
                            break
                
                # 计算占比和估值相对优化值
                if total_directions > 0:
                    adoption_ratio = adopted_directions / total_directions
                    # 计算相对优化值：(新估值-老估值)/老估值 * 100%
                    if valuation_old is not None and valuation_old > 0:
                        valuation_improvement = (valuation_new - valuation_old) / valuation_old * 100.0
                    else:
                        valuation_improvement = 0.0
                    scatter_data.append((adoption_ratio, valuation_improvement))
        
        # 存储产品数据
        products_data.append({
            'product_name': product_name,
            'product_uniq_id': product_uniq_id,
            'retail_price': retail_price,
            'optimal_price_old': optimal_price_old,
            'optimal_price_new': optimal_price_new,
            'max_total_profit_old': max_total_profit_old,
            'max_total_profit_new': max_total_profit_new,
            'groups_old': groups_old,
            'groups_new': groups_new,
            'row': row,
            'kendall_tau': kendall_tau_values[-1] if kendall_tau_values else None,
            'suggestion_analysis': suggestion_analysis,
            'optimization_table': optimization_table,
            'num_dims': num_dims,
            'scatter_data': scatter_data  # 存储该产品的散点图数据
        })
    
    # 2. 生成全局统计图表
    # 2.1 市场销售总额相对优化值分布直方图
    if total_sales_improvements:
        plt.figure(figsize=(10, 6))
        plt.hist(total_sales_improvements, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Relative Improvement (%)')
        plt.ylabel('Number of Products')
        plt.title('Distribution of Total Sales Relative Improvement')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.2, label='No Change')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'total_sales_improvement_distribution.png'))
        plt.close()
    
    # 2.2 Kendall Tau相关系数分布直方图
    if kendall_tau_values:
        plt.figure(figsize=(10, 6))
        plt.hist(kendall_tau_values, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Kendall Tau Correlation Coefficient')
        plt.ylabel('Number of Products')
        plt.title('Distribution of Kendall Tau Correlation Coefficient')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.2, label='No Correlation')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'kendall_tau_distribution.png'))
        plt.close()
    
    # 2.3 不同建议采纳次数的群体的相对优化值分布直方图（每种采纳次数一张图）
    adoption_count_chart_files = {}  # {adoption_count: filename}
    if adoption_count_to_improvements:
        adoption_counts = sorted(adoption_count_to_improvements.keys())
        
        for count in adoption_counts:
            improvements = adoption_count_to_improvements[count]
            if not improvements:
                continue
            
            # 为每个采纳次数绘制分布直方图
            plt.figure(figsize=(10, 6))
            plt.hist(improvements, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            plt.xlabel('Relative Improvement (%)')
            plt.ylabel('Number of Groups')
            plt.title(f'Distribution of Relative Improvement for Groups with {count} Adopted Suggestions')
            plt.grid(True, linestyle='--', alpha=0.5, axis='y')
            plt.axvline(x=0, color='red', linestyle='--', linewidth=1.2, label='No Change')
            
            # 添加统计信息
            avg_improvement = np.mean(improvements)
            median_improvement = np.median(improvements)
            std_improvement = np.std(improvements)
            plt.axvline(x=avg_improvement, color='green', linestyle='--', linewidth=1.2, label=f'Mean: {avg_improvement:.2f}%')
            
            # 在图上添加统计信息文本
            stats_text = f'Mean: {avg_improvement:.2f}%\nMedian: {median_improvement:.2f}%\nStd: {std_improvement:.2f}%\nCount: {len(improvements)}'
            plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)
            
            plt.legend()
            plt.tight_layout()
            
            # 保存图表
            chart_filename = f'adoption_count_{count}_improvement_distribution.png'
            chart_path = os.path.join(img_dir, chart_filename)
            plt.savefig(chart_path)
            plt.close()
            
            adoption_count_chart_files[count] = chart_filename
    
    # 2.4 优化方向采纳占比与估值优化量散点图
    all_scatter_data = []  # 收集所有产品的散点图数据
    for p_data in products_data:
        scatter_data = p_data.get('scatter_data', [])
        all_scatter_data.extend(scatter_data)
    
    if all_scatter_data:
        adoption_ratios = [x[0] for x in all_scatter_data]
        valuation_improvements = [x[1] for x in all_scatter_data]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(adoption_ratios, valuation_improvements, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.xlabel('Adoption Ratio of Optimization Directions')
        plt.ylabel('Relative Valuation Improvement (%)')
        plt.title('Relationship between Optimization Direction Adoption Ratio and Relative Valuation Improvement')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='No Change')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=1.2)

        # 设置y轴为对称对数坐标（对称双曲对数 Symlog）
        plt.yscale('symlog', linthresh=1)

        # 添加趋势线（如果数据点足够多）
        if len(all_scatter_data) > 2:
            z = np.polyfit(adoption_ratios, valuation_improvements, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(adoption_ratios), max(adoption_ratios), 100)
            plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=1.5, label='Trend Line')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'adoption_ratio_vs_valuation_improvement.png'))
        plt.close()
    
    # 2.5 产品群体划分数量与相对优化值散点图
    if group_count_vs_improvement:
        group_counts = [x[0] for x in group_count_vs_improvement]
        improvement_ratios = [x[1] for x in group_count_vs_improvement]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(group_counts, improvement_ratios, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.xlabel('Number of User Groups (Log Scale)')
        plt.ylabel('Relative Improvement (%)')
        plt.title('Relationship between Number of User Groups and Product Relative Improvement')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='No Change')
        
        # 设置x轴为对数坐标
        plt.xscale('log')
        
        # 添加趋势线（如果数据点足够多）
        if len(group_count_vs_improvement) > 2:
            # 使用对数坐标进行拟合
            log_group_counts = np.log10([max(1, gc) for gc in group_counts])  # 避免log(0)
            z = np.polyfit(log_group_counts, improvement_ratios, 1)
            p = np.poly1d(z)
            x_trend_log = np.linspace(min(log_group_counts), max(log_group_counts), 100)
            x_trend = 10 ** x_trend_log
            plt.plot(x_trend, p(x_trend_log), "r--", alpha=0.8, linewidth=1.5, label='Trend Line')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, 'group_count_vs_improvement.png'))
        plt.close()
    
    # 3. 生成Markdown报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 优化分析报告 (Optimization Analysis Report)\n\n")
        
        # 3.1 全局统计
        f.write("## 1. 全局统计 (Global Statistics)\n\n")
        
        # 优化后市场销售总额大于优化前的产品的数量占比
        total_products = len(products_data)
        if total_products > 0:
            improved_ratio = total_sales_improved_count / total_products
            f.write(f"### 1.1 优化后市场销售总额大于优化前的产品数量占比\n\n")
            f.write(f"- **改善产品数**: {total_sales_improved_count}/{total_products} ({improved_ratio:.2%})\n\n")
        
        # 平均的市场销售总额相对优化值
        if total_sales_improvements:
            avg_improvement = np.mean(total_sales_improvements)
            f.write(f"### 1.2 平均的市场销售总额相对优化值\n\n")
            f.write(f"- **平均相对优化值**: {avg_improvement:.2f}%\n")
            f.write(r"- **公式**: $\frac{1}{N}\sum_{i=1}^N\frac{new_i-old_i}{old_i} \times 100\%$")
            f.write("\n\n")
        # 市场销售总额相对优化值的分布直方图
        if total_sales_improvements:
            f.write(f"### 1.3 市场销售总额相对优化值分布直方图\n\n")
            f.write(f"![Total Sales Improvement Distribution]({img_dir_name}/total_sales_improvement_distribution.png)\n\n")
        
        # Kendall Tau相关系数统计
        if kendall_tau_values:
            avg_kendall_tau = np.mean(kendall_tau_values)
            f.write(f"### 1.4 新老群体排序Kendall Tau相关系数统计\n\n")
            f.write(f"- **平均Kendall Tau相关系数**: {avg_kendall_tau:.4f}\n")
            f.write(f"- **Kendall Tau相关系数分布直方图**:\n\n")
            f.write(f"![Kendall Tau Distribution]({img_dir_name}/kendall_tau_distribution.png)\n\n")
        
        # 不同建议采纳次数的群体的相对优化值分布
        if adoption_count_to_improvements:
            f.write(f"### 1.5 不同建议采纳次数的群体的相对优化值分布\n\n")
            f.write("该部分统计了不同建议采纳次数的群体的估值相对增量（(新估值-老估值)/老估值）的分布情况。每种采纳次数对应一张分布直方图。\n\n")
            
            adoption_counts = sorted(adoption_count_to_improvements.keys())
            for count in adoption_counts:
                improvements = adoption_count_to_improvements[count]
                if not improvements:
                    continue
                
                avg_improvement = np.mean(improvements)
                median_improvement = np.median(improvements)
                std_improvement = np.std(improvements)
                
                f.write(f"#### 采纳次数为 {count} 的群体\n\n")
                f.write(f"- **群体数量**: {len(improvements)}\n")
                f.write(f"- **平均相对优化值**: {avg_improvement:.2f}%\n")
                f.write(f"- **中位数相对优化值**: {median_improvement:.2f}%\n")
                f.write(f"- **标准差**: {std_improvement:.2f}%\n\n")
                
                chart_filename = adoption_count_chart_files.get(count)
                if chart_filename:
                    f.write(f"![Adoption Count {count} Distribution]({img_dir_name}/{chart_filename})\n\n")
                else:
                    f.write("> 图表生成失败\n\n")
        
        # 优化方向采纳占比与估值相对优化值散点图
        if all_scatter_data:
            f.write(f"### 1.6 优化方向采纳占比与估值相对优化值关系\n\n")
            f.write("该散点图展示了所有用户群体的优化方向采纳占比（该群体对应的所有优化方向中被采纳的比例）与估值相对优化值（(新估值-老估值)/老估值）的关系。\n\n")
            f.write("**说明**：\n")
            f.write("- **优化方向**：每个属性对应的取值（注意反向相关的属性需要将取值为正价值和负价值交换）\n")
            f.write("- **优化方向被采纳**：存在一个被采纳的建议，其对应属性的取值与当前用户在该属性上的取值一致\n")
            f.write("- **采纳占比**：被采纳的优化方向数 / 该用户群体对应的所有优化方向数\n")
            f.write("- **估值相对优化值**：(新估值 - 老估值) / 老估值 × 100%（相对值）\n\n")
            f.write(f"![Adoption Ratio vs Valuation Improvement]({img_dir_name}/adoption_ratio_vs_valuation_improvement.png)\n\n")
        
        # 产品群体划分数量与相对优化值散点图
        if group_count_vs_improvement:
            f.write(f"### 1.7 产品群体划分数量与相对优化值关系\n\n")
            f.write("该散点图展示了产品的群体划分数量（该产品能划分出的唯一用户群体数量）与产品的相对优化值（(新总利润-老总利润)/老总利润）的关系。\n\n")
            f.write("**说明**：\n")
            f.write("- **群体划分数量**：该产品能划分出的唯一用户群体数量（横坐标，采用对数坐标系）\n")
            f.write("- **相对优化值**：(新总利润 - 老总利润) / 老总利润 × 100%（纵坐标）\n\n")
            f.write(f"![Group Count vs Improvement]({img_dir_name}/group_count_vs_improvement.png)\n\n")
        
        f.write("---\n\n")
        
        # 3.2 逐个产品分析
        f.write("## 2. 逐个产品分析 (Product-by-Product Analysis)\n\n")
        
        # 按相对优化值升序排序
        def calculate_relative_improvement(p_data):
            """计算产品的相对优化值（用于排序）"""
            max_total_profit_old = p_data.get('max_total_profit_old', 0.0)
            max_total_profit_new = p_data.get('max_total_profit_new', 0.0)
            if max_total_profit_old > 0:
                return (max_total_profit_new - max_total_profit_old) / max_total_profit_old * 100.0
            else:
                # 如果老值为0，返回一个很小的值，确保排在前面
                return -float('inf')
        
        # 对产品按相对优化值升序排序（从小到大）
        products_data_sorted = sorted(products_data, key=calculate_relative_improvement, reverse=False)
        
        for idx, p_data in enumerate(products_data_sorted, 1):
            product_name = p_data['product_name']
            product_uniq_id = p_data['product_uniq_id']
            row = p_data['row']
            
            f.write(f"### 2.{idx} {product_name}\n\n")
            
            # 基本信息
            f.write("#### 基本信息\n\n")
            f.write(f"- **产品ID**: {product_uniq_id}\n")
            f.write(f"- **产品名称**: {product_name}\n")
            f.write(f"- **实际售价**: {p_data['retail_price']:.2f}\n")
            f.write(f"- **优化前模拟最佳定价**: {p_data['optimal_price_old']:.2f}\n")
            f.write(f"- **优化后模拟最佳定价**: {p_data['optimal_price_new']:.2f}\n")
            f.write(f"- **优化前总利润（最大总销售额）**: {p_data['max_total_profit_old']:.2f}\n")
            f.write(f"- **优化后总利润（最大总销售额）**: {p_data['max_total_profit_new']:.2f}\n")
            
            if p_data['max_total_profit_old'] > 0:
                relative_improvement = (p_data['max_total_profit_new'] - p_data['max_total_profit_old']) / p_data['max_total_profit_old'] * 100.0
                f.write(f"- **相对优化值**: {relative_improvement:.2f}%\n")
            f.write("\n")
            
            # 新的产品设计文档
            f.write("#### 新的产品设计文档\n\n")
            f.write("```markdown\n")
            hard_design = str(row.get('hard_design', ''))
            core_features = str(row.get('core_features', ''))
            value_proposition = str(row.get('value_proposition', ''))
            f.write(f"## 硬性设计\n{hard_design}\n\n")
            f.write(f"## 核心功能\n{core_features}\n\n")
            f.write(f"## 价值定位\n{value_proposition}\n")
            f.write("```\n\n")
            
            # 产品设计文档新、老的diff
            f.write("#### 产品设计文档新、老对比 (Diff)\n\n")
            f.write("```diff\n")
            hard_design_old = str(row.get('hard_design_old', ''))
            core_features_old = str(row.get('core_features_old', ''))
            value_proposition_old = str(row.get('value_proposition_old', ''))
            
            old_doc = f"## 硬性设计\n{hard_design_old}\n\n## 核心功能\n{core_features_old}\n\n## 价值定位\n{value_proposition_old}\n"
            new_doc = f"## 硬性设计\n{hard_design}\n\n## 核心功能\n{core_features}\n\n## 价值定位\n{value_proposition}\n"
            
            diff_text = generate_diff(old_doc, new_doc)
            f.write(diff_text)
            f.write("```\n\n")
            
            # 群体估值新老对比柱状图（以老分布为基准）
            if p_data['groups_old'] and p_data['groups_new']:
                img_filename_old = f"comparison_old_sort_{product_uniq_id}.png"
                plot_comparison_histogram(
                    p_data['groups_old'], p_data['groups_new'],
                    f"{product_name} - Valuation Comparison (Sorted by Old Valuation)",
                    os.path.join(img_dir, img_filename_old),
                    p_data['optimal_price_old'], p_data['optimal_price_new'],
                    sort_by_old=True,
                    suggestion_analysis=p_data.get('suggestion_analysis'),
                    optimization_table=p_data.get('optimization_table'),
                    num_dims=p_data.get('num_dims', 0)
                )
                f.write("#### 群体估值新老对比柱状图（以老分布为基准）\n\n")
                f.write(f"![Comparison Old Sort]({img_dir_name}/{img_filename_old})\n\n")
                
                # 群体估值新老对比柱状图（以新分布为基准）
                img_filename_new = f"comparison_new_sort_{product_uniq_id}.png"
                plot_comparison_histogram(
                    p_data['groups_old'], p_data['groups_new'],
                    f"{product_name} - Valuation Comparison (Sorted by New Valuation)",
                    os.path.join(img_dir, img_filename_new),
                    p_data['optimal_price_old'], p_data['optimal_price_new'],
                    sort_by_old=False,
                    suggestion_analysis=p_data.get('suggestion_analysis'),
                    optimization_table=p_data.get('optimization_table'),
                    num_dims=p_data.get('num_dims', 0)
                )
                f.write("#### 群体估值新老对比柱状图（以新分布为基准）\n\n")
                f.write(f"![Comparison New Sort]({img_dir_name}/{img_filename_new})\n\n")
            
            # 新老群体排序的Kendall Tau相关系数
            if p_data['kendall_tau'] is not None:
                f.write("#### 新老群体排序Kendall Tau相关系数\n\n")
                f.write(f"- **Kendall Tau相关系数**: {p_data['kendall_tau']:.4f}\n\n")
            
            f.write("---\n\n")
    
    print(f"Analysis complete. Report generated at {output_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyze optimization data and generate a report.")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/processed/optimization_analysis_data.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/processed/optimization_analysis.md",
        help="Path to output Markdown file"
    )
    args = parser.parse_args()

    input_csv = args.input
    output_md = args.output

    if os.path.exists(input_csv):
        analyze_optimization_data(input_csv, output_md)
    else:
        print(f"File {input_csv} not found.")

