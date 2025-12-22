import pandas as pd
import json
import re
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

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
        return 1.0 # 默认防止除零
    try:
        # 只保留数字和小数点
        clean_str = re.sub(r'[^\d.]', '', price_str)
        val = float(clean_str)
        return val if val > 0 else 1.0
    except:
        return 1.0

def plot_scatter(data_points, title, filename, color):
    """
    绘制散点图
    data_points: list of tuples (beta, avg_score)
    """
    if not data_points:
        return

    scores = [x[1] for x in data_points]      # 横坐标：平均打分
    betas = [x[0] for x in data_points]       # 纵坐标：β值

    plt.figure(figsize=(10, 6))
    plt.scatter(scores, betas, alpha=0.6, c=color)
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.2)  # 添加y=0红线
    plt.title(title)
    plt.xlabel('Average Score')
    plt.ylabel(r'$\beta$')
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_stacked_bar(distribution_data, title, filename):
    """
    绘制水平堆叠条形图
    distribution_data: dict { group_label (str/int): { score (int): count } }
    """
    if not distribution_data:
        return

    # 指定分组顺序，确保负价值、无价值、正价值为从下到上
    preferred_group_order = ['负价值', '无价值', '正价值']
    # 保留原分组中未命中的其他分组
    remaining_groups = [k for k in distribution_data.keys() if k not in preferred_group_order]
    sorted_groups = []
    for group in preferred_group_order:
        if group in distribution_data:
            sorted_groups.append(group)
    # 其余分组按字典序加入
    sorted_groups += sorted(remaining_groups)

    # 定义分数对应的颜色 (-2 到 2)
    # 使用红-黄-绿渐变
    colors = {
        -2: '#d73027', # 深红
        -1: '#fc8d59', # 橙色
        0:  '#fee08b', # 黄色
        1:  '#d9ef8b', # 浅绿
        2:  '#1a9850'  # 深绿
    }
    score_labels = [-2, -1, 0, 1, 2]
    
    # 准备数据
    labels = []
    data_matrix = {s: [] for s in score_labels}
    
    for group in sorted_groups:
        counts = distribution_data[group]
        total = sum(counts.values())
        if total == 0:
            continue
        
        labels.append(group)
        for s in score_labels:
            # 计算百分比
            pct = (counts.get(s, 0) / total) * 100
            data_matrix[s].append(pct)

    if not labels:
        return

    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 2 + len(labels) * 0.25))
    
    left_bottom = np.zeros(len(labels))
    
    for s in score_labels:
        values = data_matrix[s]
        bars = ax.barh(y_pos, values, left=left_bottom, color=colors[s], label=str(s), edgecolor='white', height=0.6)
        
        # 在条形中间添加百分比文字
        for bar, val in zip(bars, values):
            if val >= 5: # 只有占比大于5%才显示文字，避免拥挤
                width = bar.get_width()
                label_x_pos = bar.get_x() + width / 2
                label_y_pos = bar.get_y() + bar.get_height() / 2
                ax.text(label_x_pos, label_y_pos, f'{int(round(val))}%', ha='center', va='center', color='black', fontsize=9)
        
        left_bottom += np.array(values)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Percentage')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    
    # 图例
    handles, legends = ax.get_legend_handles_labels()
    # 反转图例顺序使其符合视觉逻辑（左到右 -> 上到下 或 保持顺序）
    ax.legend(handles, legends, title="Score", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def _group_value_label(v: int) -> str:
    """将群体维度取值(0/1/2)映射为中文标签。"""
    if v == 0:
        return "负价值"
    if v == 1:
        return "无价值"
    if v == 2:
        return "正价值"
    return str(v)


def _parse_group_dim_values_from_uniq_id(uniq_id: str, num_dims: int) -> list[int] | None:
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
    proportion_estimate结构来自consumer_analysis.csv中的proportion_estimate字段。
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


def plot_group_valuation_histogram(groups, title, filename, optimal_price: float):
    """
    群体估值“可变宽度柱状图”：
    - groups: list of dict { 'valuation': float, 'proportion': float }
    - x轴为累计人群占比区间，柱宽∝群体占比，柱高=估值
    - valuation>=optimal_price -> 绿色，否则灰色
    """
    if not groups:
        return
    # 基础宽度：保证极小占比群体也有足够空间标注（基础宽度尽量小）
    n = len(groups)
    base_width = min(0.01, 0.25 / max(1, n))

    widths = np.array([max(0.0, float(g.get("proportion", 0.0))) for g in groups], dtype=float)
    heights = np.array([float(g.get("valuation", 0.0)) for g in groups], dtype=float)
    total_w = float(widths.sum())
    if total_w <= 0:
        # 退化处理：等宽
        widths = np.ones_like(widths) / max(1, len(widths))
        total_w = float(widths.sum())
    else:
        # 在“占比宽度”之外叠加基础宽度，再整体归一化
        widths = widths + base_width
        widths = widths / float(widths.sum())

    lefts = np.concatenate([[0.0], np.cumsum(widths)[:-1]])
    colors = ["#1a9850" if h >= float(optimal_price) else "#bdbdbd" for h in heights]

    plt.figure(figsize=(12, 5))
    plt.bar(lefts, heights, width=widths, align="edge", color=colors, edgecolor="white", linewidth=0.2)
    plt.axhline(y=float(optimal_price), color="red", linestyle="--", linewidth=1.2, label="Optimal Price")
    plt.title(title)
    plt.xlabel("Cumulative Group Proportion")
    plt.ylabel("Valuation (Psychological Price)")
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")

    # 标注：用连续的M个数字表示该群体取值（例如00120）
    max_h = float(np.max(heights)) if len(heights) else 0.0
    y_offset = max(1.0, 0.01 * max_h)
    for g, left, w, h in zip(groups, lefts, widths, heights):
        code = str(g.get("dim_code", "") or "")
        if not code:
            continue
        x = float(left) + float(w) / 2.0
        y = float(h) + y_offset
        plt.text(x, y, code, rotation=45, ha="right", va="bottom", fontsize=6, color="black")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def analyze_consumer_data(input_file, output_file):
    # 确保图片输出目录存在
    img_dir = os.path.join(os.path.dirname(output_file), 'pictures')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # 存储所有产品的统计信息
    products_data = {}

    # 1. 遍历每一行数据，构建产品和属性的统计映射
    for index, row in df.iterrows():
        product_name = row['modified_name']
        product_uniq_id = row['product_uniq_id']
        beta_plus_coeffs = safe_json_loads(row.get('beta_plus_coefficients'))
        beta_minus_coeffs = safe_json_loads(row.get('beta_minus_coefficients'))
        segmentation = safe_json_loads(row['segmentation_result'])
        attr_analysis = safe_json_loads(row['attribute_analysis'])
        k_attr_scores = safe_json_loads(row.get('k_attr_scores'))
        uniq_id = str(row['uniq_id'])
        price_str = row.get('retail_price', '1') # 获取价格
        beta_0 = row.get('beta_0')
        optimal_price = row.get('optimal_price')
        max_total_profit = row.get('max_total_profit')
        center_group_ids = safe_json_loads(row.get('optimal_price_center_group_ids'))
        proportion_estimate = safe_json_loads(row.get('proportion_estimate'))

        if (not product_name
            or beta_plus_coeffs is None
            or beta_minus_coeffs is None
            or segmentation is None
            or attr_analysis is None):
            continue
        if (not isinstance(beta_plus_coeffs, list)
            or not isinstance(beta_minus_coeffs, list)
            or len(beta_plus_coeffs) != len(beta_minus_coeffs)):
            continue

        # 初始化产品数据结构
        if product_name not in products_data:
            price = parse_price(price_str)
            beta_plus = [float(b) for b in beta_plus_coeffs]
            beta_minus = [float(b) for b in beta_minus_coeffs]
            # 负β个数：按β^-为负的维度计数（后续统计会进一步剔除“不可识别”的维度）
            neg_beta_count = sum(1 for b in beta_minus if b < 0)
            
            # 构建属性到维度的映射表
            attr_map = {}
            dims = segmentation.get('segmentations', [])
            
            # 提取产品所有属性的描述备用
            attr_desc_map = {}
            for item in segmentation.get('total_attributes', []):
                attr_desc_map[item['attribute_name']] = item['attribute_description']

            for idx, dim in enumerate(dims):
                dim_name = dim.get('dimension_name', f"Dimension {idx+1}")
                beta_plus_val = beta_plus[idx] if idx < len(beta_plus) else 0.0
                beta_minus_val = beta_minus[idx] if idx < len(beta_minus) else 0.0
                
                # 正向属性
                for attr in dim.get('positive_attribute_names', []):
                    attr_map[attr] = {
                        'dim_index': idx,
                        'dim_name': dim_name,
                        'beta_plus': beta_plus_val,
                        'beta_minus': beta_minus_val,
                        'direction': '正向'
                    }
                # 反向属性
                for attr in dim.get('negative_attribute_names', []):
                    attr_map[attr] = {
                        'dim_index': idx,
                        'dim_name': dim_name,
                        'beta_plus': beta_plus_val,
                        'beta_minus': beta_minus_val,
                        'direction': '反向'
                    }

            products_data[product_name] = {
                'product_uniq_id': product_uniq_id,
                'beta_0': beta_0,
                'beta_plus': beta_plus,
                'beta_minus': beta_minus,
                'price': price,
                'neg_beta_count': neg_beta_count,
                'optimal_price': optimal_price,
                'max_total_profit': max_total_profit,
                'center_group_ids': center_group_ids if isinstance(center_group_ids, list) else [],
                'proportion_estimate': proportion_estimate if isinstance(proportion_estimate, list) else None,
                'doc_fields': {
                    'hard_design': row.get('hard_design', ''),
                    'core_features': row.get('core_features', ''),
                    'value_proposition': row.get('value_proposition', ''),
                },
                'attr_map': attr_map,
                'segmentation_data': segmentation,
                'attr_desc_map': attr_desc_map,
                'stats': {},
                'k_scores': k_attr_scores if isinstance(k_attr_scores, dict) else {},
                # 维度取值覆盖情况：用于判断某侧β是否可识别（该维度是否存在取值2/0的群体）
                'dim_has_pos_value': [False] * len(beta_plus),  # 是否存在正价值(2)群体
                'dim_has_neg_value': [False] * len(beta_plus),  # 是否存在负价值(0)群体
            }

        num_dims = len(beta_plus_coeffs)
        parts = uniq_id.split('_')
        
        # 提取用户维度取值
        if len(parts) >= num_dims:
            try:
                user_dim_values = [int(x) for x in parts[-num_dims:]]
            except ValueError:
                continue
        else:
            continue

        # 更新“该产品各维度是否存在某种取值的群体”的统计
        try:
            p_data = products_data[product_name]
            for i, v in enumerate(user_dim_values):
                if v == 2 and i < len(p_data['dim_has_pos_value']):
                    p_data['dim_has_pos_value'][i] = True
                if v == 0 and i < len(p_data['dim_has_neg_value']):
                    p_data['dim_has_neg_value'][i] = True
        except Exception:
            pass

        # 统计该用户的属性打分
        p_data = products_data[product_name]
        attr_map = p_data['attr_map']
        
        for attr_item in attr_analysis:
            attr_name = attr_item.get('attribute_name')
            actual_score = attr_item.get('attribute_score')
            
            if attr_name in attr_map:
                info = attr_map[attr_name]
                dim_idx = info['dim_index']
                direction = info['direction']
                
                if dim_idx < len(user_dim_values):
                    user_val_for_dim = user_dim_values[dim_idx] # 0, 1, or 2
                    
                    if attr_name not in p_data['stats']:
                        p_data['stats'][attr_name] = {
                            0: {'correct': 0, 'total': 0, 'score_sum': 0, 'score_counts': {-2:0, -1:0, 0:0, 1:0, 2:0}}, 
                            1: {'correct': 0, 'total': 0, 'score_sum': 0, 'score_counts': {-2:0, -1:0, 0:0, 1:0, 2:0}}, 
                            2: {'correct': 0, 'total': 0, 'score_sum': 0, 'score_counts': {-2:0, -1:0, 0:0, 1:0, 2:0}}
                        }
                    
                    stat_entry = p_data['stats'][attr_name][user_val_for_dim]
                    stat_entry['total'] += 1
                    stat_entry['score_sum'] += actual_score # 累加分数
                    
                    # 记录具体的分布，防止KeyError（虽然理论上分数应该在-2到2之间）
                    safe_score = int(actual_score)
                    if safe_score in stat_entry['score_counts']:
                        stat_entry['score_counts'][safe_score] += 1
                    
                    is_match = False
                    if direction == '正向':
                        if user_val_for_dim == 0 and actual_score < 0: is_match = True
                        elif user_val_for_dim == 1 and actual_score == 0: is_match = True
                        elif user_val_for_dim == 2 and actual_score > 0: is_match = True
                    elif direction == '反向':
                        if user_val_for_dim == 0 and actual_score > 0: is_match = True
                        elif user_val_for_dim == 1 and actual_score == 0: is_match = True
                        elif user_val_for_dim == 2 and actual_score < 0: is_match = True
                    
                    if is_match:
                        stat_entry['correct'] += 1

    # 2. 计算全局统计、k比例分布以及绘图数据
    global_stats = {
        0: {'correct': 0, 'total': 0},
        1: {'correct': 0, 'total': 0},
        2: {'correct': 0, 'total': 0}
    }

    # k值为正比例 -> β方向计数（双β：分别统计β^+与β^-）
    k_ratio_buckets_plus = {}  # ratio_str -> {'beta_pos': cnt, 'beta_neg': cnt}
    k_ratio_buckets_neg = {}   # ratio_str -> {'beta_pos': cnt, 'beta_neg': cnt}
    
    # 绘图数据 (beta, average_score)
    plot_points = {
        'neg': [], # 负价值群体平均分 vs β^-
        'pos': []  # 正价值群体平均分 vs β^+
    }

    for p_name, p_data in products_data.items():
        has_pos = p_data.get('dim_has_pos_value', []) or []
        has_neg = p_data.get('dim_has_neg_value', []) or []
        beta_plus_list = p_data.get('beta_plus', []) or []
        beta_minus_list = p_data.get('beta_minus', []) or []

        for attr_name, stats in p_data['stats'].items():
            info = p_data['attr_map'][attr_name]
            beta_minus = info.get('beta_minus', 0.0)  # β^-
            beta_plus = info.get('beta_plus', 0.0)    # β^+
            direction = info['direction']
            k_scores_map = p_data.get('k_scores', {})
            
            # 汇总全局数据（仍然保持 accuracy 统计用于文本报告）
            for i in range(3):
                global_stats[i]['correct'] += stats[i]['correct']
                global_stats[i]['total'] += stats[i]['total']
            
            # 准备绘图数据
            # 负价值图表：只统计“负价值群体平均分 vs β^-”
            # 正向相关 -> 取Dim取值0
            # 反向相关 -> 取Dim取值2
            target_idx = 0 if direction == '正向' else 2
            if stats[target_idx]['total'] > 0:
                avg_score = stats[target_idx]['score_sum'] / stats[target_idx]['total']
                plot_points['neg'].append((beta_minus, avg_score))

            # 正价值图表：只统计“正价值群体平均分 vs β^+”
            # 正向相关 -> 取Dim取值2
            # 反向相关 -> 取Dim取值0
            target_idx = 2 if direction == '正向' else 0
            if stats[target_idx]['total'] > 0:
                avg_score = stats[target_idx]['score_sum'] / stats[target_idx]['total']
                plot_points['pos'].append((beta_plus, avg_score))

        # 计算维度层面的“k为正”比例分布（分别对应β^+与β^-）
        k_scores_map = p_data.get('k_scores', {})
        dims = p_data['segmentation_data'].get('segmentations', [])
        for idx, dim in enumerate(dims):
            attrs = dim.get('positive_attribute_names', []) + dim.get('negative_attribute_names', [])
            valid_attrs = [
                a for a in attrs
                if a in k_scores_map and isinstance(k_scores_map.get(a), dict)
            ]
            if not valid_attrs:
                continue

            # k_plus>0比例 vs β+ 方向
            pos_k_plus_cnt = sum(1 for a in valid_attrs if float(k_scores_map[a].get('k_plus', 0.0)) > 0)
            ratio_plus = pos_k_plus_cnt / len(valid_attrs)
            ratio_plus_key = f"{ratio_plus:.2f}"
            beta_plus_val = beta_plus_list[idx] if idx < len(beta_plus_list) else 0.0
            bucket_plus = k_ratio_buckets_plus.setdefault(ratio_plus_key, {'beta_pos': 0, 'beta_neg': 0})
            # 对于“算不出β^+”的维度（不存在取值2群体），不计入统计；同时β^+=0也不计入，避免比例不为1
            if idx < len(has_pos) and has_pos[idx] and abs(float(beta_plus_val)) > 1e-12:
                if beta_plus_val > 0:
                    bucket_plus['beta_pos'] += 1
                elif beta_plus_val < 0:
                    bucket_plus['beta_neg'] += 1

            # k_minus>0比例 vs β^- 方向
            pos_k_minus_cnt = sum(1 for a in valid_attrs if float(k_scores_map[a].get('k_minus', 0.0)) > 0)
            ratio_neg = pos_k_minus_cnt / len(valid_attrs)
            ratio_neg_key = f"{ratio_neg:.2f}"
            beta_minus_val = beta_minus_list[idx] if idx < len(beta_minus_list) else 0.0
            bucket_neg = k_ratio_buckets_neg.setdefault(ratio_neg_key, {'beta_pos': 0, 'beta_neg': 0})
            # 对于“算不出β^-”的维度（不存在取值0群体），不计入统计；同时β^-=0也不计入
            if idx < len(has_neg) and has_neg[idx] and abs(float(beta_minus_val)) > 1e-12:
                if beta_minus_val > 0:
                    bucket_neg['beta_pos'] += 1
                elif beta_minus_val < 0:
                    bucket_neg['beta_neg'] += 1

    # 生成图表：散点图（按要求仅保留：负价值-$\beta^-$、正价值-$\beta^+$）
    plot_scatter(plot_points['neg'], r'Negative Value Avg Score vs $\beta^-$', os.path.join(img_dir, 'neg_value_avg_score_vs_beta_minus.png'), 'red')
    plot_scatter(plot_points['pos'], r'Positive Value Avg Score vs $\beta^+$', os.path.join(img_dir, 'pos_value_avg_score_vs_beta_plus.png'), 'green')

    # 生成k比例与β方向分布的水平堆叠图
    def plot_k_ratio_bucket(bucket_stats, filename, title):
        if not bucket_stats:
            return
        ratios_sorted = sorted(bucket_stats.keys(), key=lambda x: float(x))
        y_labels = []
        pos_vals = []
        neg_vals = []
        attr_counts = []  # 存储每个分组的属性个数
        
        for key in ratios_sorted:
            data = bucket_stats[key]
            total = data['beta_pos'] + data['beta_neg']
            if total == 0:
                continue
            y_labels.append(key)
            pos_vals.append(data['beta_pos'] / total * 100)
            neg_vals.append(data['beta_neg'] / total * 100)
            attr_counts.append(total)  # total 就是该组的属性个数（维度个数）
        
        if not y_labels:
            return
        
        # 计算总属性个数和每个柱子的高度占比
        total_attrs = sum(attr_counts)
        if total_attrs == 0:
            return
        
        # 基础高度（归一化后的最大高度）
        base_height = 4
        # 固定间距
        spacing = 0.05
        
        # 计算每个柱子的高度（按属性个数占比）
        bar_heights = [base_height * (count / total_attrs) + 0.05 for count in attr_counts]
        
        # 计算每个柱子的y位置（从底部开始，等距间距）
        y_positions = []
        current_y = 0
        for height in bar_heights:
            y_positions.append(current_y + height / 2)  # 柱子中心位置
            current_y += height + spacing  # 下一个柱子的起始位置
        
        # 计算总高度（用于设置图形大小）
        total_height = current_y - spacing  # 减去最后一个间距
        
        # 创建图形，高度设为原来的2倍以确保最细的柱子也能正常显示
        fig_height = max(4, total_height * 2)  # 至少4，或总高度的2倍
        fig, ax = plt.subplots(figsize=(12, fig_height))
        
        # 绘制水平堆叠条形图
        bars_neg = ax.barh(y_positions, neg_vals, height=bar_heights, 
                          color='#d73027', label='β<0', edgecolor='white')
        bars_pos = ax.barh(y_positions, pos_vals, left=neg_vals, height=bar_heights,
                          color='#1a9850', label='β>0', edgecolor='white')
        
        # 在柱内部标注β正/负占比（如果宽度足够）
        for bars in (bars_neg, bars_pos):
            for bar in bars:
                width = bar.get_width()
                if width >= 5:
                    ax.text(bar.get_x() + width / 2, bar.get_y() + bar.get_height() / 2,
                            f"{width:.0f}%", ha='center', va='center', color='black', fontsize=9)
        
        # 在柱外部（右侧）标注属性个数占比
        attr_ratios = [count / total_attrs * 100 for count in attr_counts]
        for i, (y_pos, ratio) in enumerate(zip(y_positions, attr_ratios)):
            # 计算柱的右边界位置
            right_edge = neg_vals[i] + pos_vals[i]
            # 在柱右侧标注占比百分比
            ax.text(right_edge + 3, y_pos, f"{ratio:.1f}%", ha='left', va='center', 
                   color='black', fontsize=9, weight='bold')
        
        # 设置y轴刻度和标签
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('占比(%)')
        ax.set_title(title)
        
        # 调整x轴范围，为外部标注留出空间
        max_width = max(neg_vals[i] + pos_vals[i] for i in range(len(y_labels)))
        ax.set_xlim(0, max(100, max_width + 15))  # 留出15%的空间用于标注
        
        # 设置y轴范围
        ax.set_ylim(-spacing, current_y)
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    k_ratio_plus_chart = os.path.join(img_dir, 'k_ratio_beta_plus_distribution.png')
    plot_k_ratio_bucket(
        k_ratio_buckets_plus,
        k_ratio_plus_chart,
        r'各维度$k^+>0$比例分组的$\beta^+$正/负占比（柱高=维度个数占比）'
    )

    k_ratio_neg_chart = os.path.join(img_dir, 'k_ratio_beta_neg_distribution.png')
    plot_k_ratio_bucket(
        k_ratio_buckets_neg,
        k_ratio_neg_chart,
        r'各维度$k^->0$比例分组的$\beta^-$正/负占比（柱高=维度个数占比）'
    )

    # 计算总准确率
    total_correct_all = sum(g['correct'] for g in global_stats.values())
    total_count_all = sum(g['total'] for g in global_stats.values())

    # 计算总体β方向占比（所有产品、所有维度）：分别统计β^+与β^-，并给出合并统计
    # 口径：
    # - 若某维度不存在取值2群体，则该维度β^+不可识别 -> 不计入β^+统计
    # - 若某维度不存在取值0群体，则该维度β^-不可识别 -> 不计入β^-统计
    # - 同时将β==0的维度也不计入（否则正/负占比之和可能不为1）
    beta_plus_pos = beta_plus_neg = 0
    beta_minus_pos = beta_minus_neg = 0

    for p_data in products_data.values():
        has_pos = p_data.get('dim_has_pos_value', []) or []
        has_neg = p_data.get('dim_has_neg_value', []) or []
        beta_plus_list = p_data.get('beta_plus', []) or []
        beta_minus_list = p_data.get('beta_minus', []) or []

        for i, b in enumerate(beta_plus_list):
            if i < len(has_pos) and has_pos[i] and isinstance(b, (int, float)) and abs(float(b)) > 1e-12:
                if b > 0:
                    beta_plus_pos += 1
                elif b < 0:
                    beta_plus_neg += 1

        for i, b in enumerate(beta_minus_list):
            if i < len(has_neg) and has_neg[i] and isinstance(b, (int, float)) and abs(float(b)) > 1e-12:
                if b > 0:
                    beta_minus_pos += 1
                elif b < 0:
                    beta_minus_neg += 1

    beta_plus_total = beta_plus_pos + beta_plus_neg
    beta_minus_total = beta_minus_pos + beta_minus_neg
    beta_merged_total = beta_plus_total + beta_minus_total
    beta_merged_pos = beta_plus_pos + beta_minus_pos
    beta_merged_neg = beta_plus_neg + beta_minus_neg
    
    # 3. 生成Markdown报告
    sorted_products = sorted(
        products_data.items(), 
        key=lambda item: item[1]['neg_beta_count'], 
        reverse=True
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 消费者分析报告 (Consumer Analysis Report)\n\n")
        
        # 3.1 全局统计
        f.write("## 1. 全局统计 (Global Statistics)\n\n")
        f.write(f"- **负价值群体总准确率**: {global_stats[0]['correct']}/{global_stats[0]['total']} ({global_stats[0]['correct']/global_stats[0]['total']:.2%})\n")
        f.write(f"- **无价值群体总准确率**: {global_stats[1]['correct']}/{global_stats[1]['total']} ({global_stats[1]['correct']/global_stats[1]['total']:.2%})\n")
        f.write(f"- **正价值群体总准确率**: {global_stats[2]['correct']}/{global_stats[2]['total']} ({global_stats[2]['correct']/global_stats[2]['total']:.2%})\n")
        f.write(f"- **总和准确率**: {total_correct_all}/{total_count_all} ({total_correct_all/total_count_all:.2%})\n\n")

        # β方向统计（分别+合并）
        f.write("### $\\beta$方向统计（双$\\beta$）\n\n")
        if beta_plus_total > 0:
            f.write(f"- **$\\beta^+$ 为正占比**: {beta_plus_pos}/{beta_plus_total} ({beta_plus_pos/beta_plus_total:.2%})；为负占比: {beta_plus_neg}/{beta_plus_total} ({beta_plus_neg/beta_plus_total:.2%})\n")
        else:
            f.write("- **$\\beta^+$ 为正/负占比**: 无有效$\\beta^+$数据\n")

        if beta_minus_total > 0:
            f.write(f"- **$\\beta^-$ 为正占比**: {beta_minus_pos}/{beta_minus_total} ({beta_minus_pos/beta_minus_total:.2%})；为负占比: {beta_minus_neg}/{beta_minus_total} ({beta_minus_neg/beta_minus_total:.2%})\n")
        else:
            f.write("- **$\\beta^-$ 为正/负占比**: 无有效$\\beta^-$数据\n")

        if beta_merged_total > 0:
            f.write(f"- **合并($\\beta^+$ 与 $\\beta^-$) 为正占比**: {beta_merged_pos}/{beta_merged_total} ({beta_merged_pos/beta_merged_total:.2%})；为负占比: {beta_merged_neg}/{beta_merged_total} ({beta_merged_neg/beta_merged_total:.2%})\n\n")
        else:
            f.write("- **合并($\\beta^+$ 与 $\\beta^-$) 为正/负占比**: 无有效$\\beta$数据\n\n")

        f.write("### k为正比例分布下的β方向占比（双β）\n\n")
        if k_ratio_buckets_plus:
            f.write("#### k_plus为正比例分布下的β+方向占比\n\n")
            f.write("![k_ratio_beta_plus_distribution](pictures/k_ratio_beta_plus_distribution.png)\n\n")
        else:
            f.write("> 无可用数据生成 β+ 的k比例分布图\n\n")

        if k_ratio_buckets_neg:
            f.write("#### $k^-$为正比例分布下的$\\beta^-$方向占比\n\n")
            f.write("![k_ratio_beta_neg_distribution](pictures/k_ratio_beta_neg_distribution.png)\n\n")
        else:
            f.write("> 无可用数据生成 $\\beta^-$ 的k比例分布图\n\n")
        
        f.write("## 2. 平均分与$\\beta$散点图 (Avg Score vs $\\beta$)\n\n")
        f.write("> **注**: 这里只统计：负价值群体平均分 vs $\\beta^-$；正价值群体平均分 vs $\\beta^+$\n\n")
        f.write("### 负价值群体平均分 vs $\\beta^-$\n")
        f.write("![Negative Value Group Avg Score vs Beta_Minus](pictures/neg_value_avg_score_vs_beta_minus.png)\n\n")
        f.write("### 正价值群体平均分 vs $\\beta^+$\n")
        f.write("![Positive Value Group Avg Score vs Beta_Plus](pictures/pos_value_avg_score_vs_beta_plus.png)\n\n")

        f.write("## 3. 产品详细分析 (Product Details)\n\n")
        
        for product_name, data in sorted_products:
            f.write(f"### Product: {product_name}\n")
            f.write(f"**ID**: {data['product_uniq_id']}\n")
            f.write(f"**实际售价**: {data['price']}\n")
            f.write(f"**模拟最佳定价** ($p^*$): {data.get('optimal_price')}\n")
            f.write(f"**总利润（最大总销售额）** ($\\Pi$): {data.get('max_total_profit')}\n")
            f.write(f"**截距** ($\\beta_0$): {data.get('beta_0')}\n\n")

            # 产品完整文档
            f.write("#### 产品完整文档\n\n")
            docs = data.get('doc_fields', {}) or {}
            hard_design = str(docs.get('hard_design', '') or '').strip()
            core_features = str(docs.get('core_features', '') or '').strip()
            value_prop = str(docs.get('value_proposition', '') or '').strip()
            f.write("```markdown\n")
            f.write("## hard_design\n\n")
            f.write(f"{hard_design}\n\n")
            f.write("## core_features\n\n")
            f.write(f"{core_features}\n\n")
            f.write("## value_proposition\n\n")
            f.write(f"{value_prop}\n")
            f.write("```\n\n")

            f.write("#### 双$\\beta$参数\n\n")
            f.write(f"- **$\\beta^+$**: {data.get('beta_plus')}\n")
            f.write(f"- **$\\beta^-$**: {data.get('beta_minus')}\n\n")

            # 群体估值柱状图
            f.write("#### 产品的群体估值柱状图\n\n")
            sub_df = df[df['product_uniq_id'] == data['product_uniq_id']]
            num_dims = len(data.get('beta_plus', []) or [])
            optimal_price_val = float(data.get('optimal_price') or 0.0)
            prop_est = data.get('proportion_estimate')
            groups_list = []
            if not sub_df.empty and isinstance(prop_est, list) and num_dims > 0:
                for gid, gdf in sub_df.groupby('uniq_id'):
                    dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
                    if dim_vals is None:
                        continue
                    vals = [parse_price(x) for x in gdf['psychological_price'].tolist()]
                    valuation = float(np.mean(vals)) if vals else 0.0
                    proportion = _calc_group_proportion(prop_est, dim_vals)
                    dim_code = "".join(str(int(v)) for v in dim_vals)
                    groups_list.append({'uniq_id': str(gid), 'valuation': valuation, 'proportion': proportion, 'dim_vals': dim_vals, 'dim_code': dim_code})
                groups_list.sort(key=lambda x: x['valuation'], reverse=True)
                chart_filename = f"valuation_{data['product_uniq_id']}.png"
                chart_path = os.path.join(img_dir, chart_filename)
                plot_group_valuation_histogram(
                    [{'valuation': g['valuation'], 'proportion': g['proportion'], 'dim_code': g.get('dim_code', '')} for g in groups_list],
                    f"Group Valuations for {product_name}",
                    chart_path,
                    optimal_price_val
                )
                f.write(f"![Group Valuations](pictures/{chart_filename})\n\n")
            else:
                f.write("> 无可用数据生成群体估值柱状图\n\n")
            
            # 属性统计表格
            f.write("#### 属性统计总表\n\n")
            f.write("| 属性名称 | 对应维度 | 对应方向 | $\\beta^+$ | $\\beta^-$ | 负价值准确率 | 无价值准确率 | 正价值准确率 | 总准确率 | $k^+$ | $k^-$ |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            attr_names = sorted(data['stats'].keys(), key=lambda x: data['attr_map'][x]['dim_index'])
            k_scores = data.get('k_scores', {})
            
            for attr_name in attr_names:
                info = data['attr_map'][attr_name]
                stats = data['stats'][attr_name]
                
                def fmt_acc(grp_idx):
                    s = stats[grp_idx]
                    if s['total'] == 0:
                        return "-/-"
                    return f"{s['correct']}/{s['total']}"
                
                neg_acc_str = fmt_acc(0)
                neu_acc_str = fmt_acc(1)
                pos_acc_str = fmt_acc(2)
                
                total_correct = stats[0]['correct'] + stats[1]['correct'] + stats[2]['correct']
                total_count = stats[0]['total'] + stats[1]['total'] + stats[2]['total']
                total_acc_str = f"{total_correct}/{total_count}"
                
                # 获取k+/k-并格式化
                k_val = k_scores.get(attr_name)
                k_plus = k_minus = None
                if isinstance(k_val, dict):
                    k_plus = k_val.get('k_plus')
                    k_minus = k_val.get('k_minus')

                def fmt_k(v):
                    if not isinstance(v, (int, float)):
                        return "-"
                    s = f"{float(v):.4f}"
                    return f"=={s}==" if float(v) < 0 else s

                f.write(
                    f"| {attr_name} | {info['dim_name']} | {info['direction']} | {info.get('beta_plus', 0.0):.4f} | {info.get('beta_minus', 0.0):.4f} | "
                    f"{neg_acc_str} | {neu_acc_str} | {pos_acc_str} | {total_acc_str} | {fmt_k(k_plus)} | {fmt_k(k_minus)} |\n"
                )
            
            # 添加潜在用户群体定义
            user_group_def = data['segmentation_data'].get('user_group', 'N/A')
            f.write(f"\n**潜在用户群体定义**: \n{user_group_def}\n\n")

            # 所有维度的详细信息（不再只显示负β）
            f.write("#### 维度详细分析（全部维度）\n\n")
            dims = data['segmentation_data'].get('segmentations', [])
            for idx, dim_info in enumerate(dims):
                dim_name = dim_info.get('dimension_name', f"Dimension {idx+1}")
                beta_plus_val = (data.get('beta_plus', []) or [])[idx] if idx < len((data.get('beta_plus', []) or [])) else 0.0
                beta_minus_val = (data.get('beta_minus', []) or [])[idx] if idx < len((data.get('beta_minus', []) or [])) else 0.0
                f.write(f"**维度**: {dim_name} ($\\beta^+\\,={beta_plus_val:.4f}$, $\\beta^-\\,={beta_minus_val:.4f}$)\n\n")

                # 列出相关属性及其详细信息（含打分分布图）
                pos_attrs = dim_info.get('positive_attribute_names', [])
                neg_attrs = dim_info.get('negative_attribute_names', [])
                all_related_attrs = pos_attrs + neg_attrs
                if all_related_attrs:
                    f.write("- **相关属性与建议/打分分布**:\n")
                    for attr in all_related_attrs:
                        direction = "正向" if attr in pos_attrs else "反向"
                        desc = data['attr_desc_map'].get(attr, "无描述")
                        k_val = data.get('k_scores', {}).get(attr)
                        k_plus = k_minus = None
                        if isinstance(k_val, dict):
                            k_plus = k_val.get('k_plus')
                            k_minus = k_val.get('k_minus')
                        k_plus_str = f"{float(k_plus):.4f}" if isinstance(k_plus, (int, float)) else "-"
                        k_minus_str = f"{float(k_minus):.4f}" if isinstance(k_minus, (int, float)) else "-"
                        f.write(f"  - **{attr}** ({direction}, k+={k_plus_str}, k-={k_minus_str}): {desc}\n")

                        if attr in data['stats']:
                            attr_stats = data['stats'][attr]
                            distribution_data = {
                                '负价值': attr_stats[0]['score_counts'],
                                '无价值': attr_stats[1]['score_counts'],
                                '正价值': attr_stats[2]['score_counts']
                            }
                            safe_attr_name = re.sub(r'\W+', '_', attr)
                            chart_filename = f"dist_{data['product_uniq_id']}_d{idx}_{safe_attr_name}.png"
                            chart_path = os.path.join(img_dir, chart_filename)
                            plot_stacked_bar(
                                distribution_data,
                                f"Score Distribution for '{attr}'",
                                chart_path
                            )
                            f.write(f"\n    ![Score Distribution for {attr}](pictures/{chart_filename})\n\n")

                # 维度取值判定标准及证明
                f.write("- **取值判定标准及合理性**:\n")
                if dim_info.get('negative_value_criteria'):
                    f.write(f"  - **负价值**: {dim_info['negative_value_criteria']}\n")
                if dim_info.get('negative_value_rationale'):
                    f.write(f"    - *证明*: {dim_info['negative_value_rationale']}\n")
                if dim_info.get('neutral_value_criteria'):
                    f.write(f"  - **无价值**: {dim_info['neutral_value_criteria']}\n")
                if dim_info.get('neutral_value_rationale'):
                    f.write(f"    - *证明*: {dim_info['neutral_value_rationale']}\n")
                if dim_info.get('positive_value_criteria'):
                    f.write(f"  - **正价值**: {dim_info['positive_value_criteria']}\n")
                if dim_info.get('positive_value_rationale'):
                    f.write(f"    - *证明*: {dim_info['positive_value_rationale']}\n")
                f.write("\n")

            # 中心群体的群体定义和优化建议（来自consumer_analysis.csv已计算的中心群体集合）
            f.write("#### 中心群体（与最佳定价估值相等的群体）\n\n")
            center_ids = data.get('center_group_ids', []) or []
            # 建立uniq_id -> 单行记录，用于读取该群体的优化建议
            sub_df = df[df['product_uniq_id'] == data['product_uniq_id']]
            num_dims = len(data.get('beta_plus', []) or [])
            prop_est = data.get('proportion_estimate')
            
            if not center_ids:
                f.write("> 未找到中心群体ID\n\n")
            else:
                
                # 计算每个中心群体的比例，并按比例降序排序
                center_groups_with_prop = []
                for gid in center_ids:
                    dim_vals = _parse_group_dim_values_from_uniq_id(str(gid), num_dims) if num_dims > 0 else None
                    if dim_vals is not None and isinstance(prop_est, list):
                        proportion = _calc_group_proportion(prop_est, dim_vals)
                        center_groups_with_prop.append((gid, proportion, dim_vals))
                    else:
                        center_groups_with_prop.append((gid, 0.0, dim_vals))
                
                # 按比例降序排序
                center_groups_with_prop.sort(key=lambda x: x[1], reverse=True)
                
                for gid, proportion, dim_vals in center_groups_with_prop:
                    f.write(f"##### 群体: `{gid}` (占比: {proportion:.4f})\n\n")

                    dims = data['segmentation_data'].get('segmentations', [])
                    if dim_vals is not None and dims and len(dim_vals) == len(dims):
                        f.write("**群体定义（各维度取值）**:\n\n")
                        for i, v in enumerate(dim_vals):
                            dim_name = dims[i].get('dimension_name', f"Dimension {i+1}")
                            v_label = _group_value_label(v)
                            # 对应判定标准
                            criteria_key = 'neutral_value_criteria'
                            if v == 0:
                                criteria_key = 'negative_value_criteria'
                            elif v == 2:
                                criteria_key = 'positive_value_criteria'
                            criteria = dims[i].get(criteria_key, '')
                            if criteria:
                                f.write(f"- **{dim_name}**: {v_label}（标准：{criteria}）\n")
                            else:
                                f.write(f"- **{dim_name}**: {v_label}\n")
                        f.write("\n")
                    else:
                        f.write("> 无法解析该群体维度取值\n\n")

                    matched = sub_df[sub_df['uniq_id'].astype(str) == str(gid)]
                    if matched.empty:
                        f.write("> 未找到该群体对应的attribute_analysis\n\n")
                        continue

                    row0 = matched.iloc[0]
                    aa = safe_json_loads(row0.get('attribute_analysis'))
                    if not isinstance(aa, list):
                        f.write("> attribute_analysis解析失败\n\n")
                        continue

                    # 收集所有属性建议，包含打分和维度取值信息
                    attr_suggestions = []
                    for item in aa:
                        if not isinstance(item, dict):
                            continue
                        attr_name = item.get('attribute_name')
                        sug = item.get('attribute_optimization_suggestion', '')
                        score = item.get('attribute_score', 0)
                        effect = item.get('attribute_effect', '')
                        if not attr_name:
                            continue
                        
                        # 获取该属性对应的维度取值
                        dim_idx = data['attr_map'].get(attr_name, {}).get('dim_index')
                        dim_val_for_attr = None
                        dim_name_for_attr = None
                        if isinstance(dim_idx, int) and dim_vals is not None and 0 <= dim_idx < len(dim_vals):
                            dim_val_for_attr = dim_vals[dim_idx]
                            if dim_idx < len(dims):
                                dim_name_for_attr = dims[dim_idx].get('dimension_name', f"Dimension {dim_idx+1}")
                        
                        attr_suggestions.append({
                            'attr_name': attr_name,
                            'suggestion': sug,
                            'score': score,
                            'effect': effect,
                            'dim_val': dim_val_for_attr,
                            'dim_name': dim_name_for_attr
                        })
                    
                    # 按满意度打分升序排序（打分低的排在前面）
                    attr_suggestions.sort(key=lambda x: x['score'], reverse=False)
                    
                    # 显示所有建议
                    if attr_suggestions:
                        f.write("**优化建议（按满意度打分升序排列）**:\n\n")
                        for attr_item in attr_suggestions:
                            attr_name = attr_item['attr_name']
                            sug = str(attr_item['suggestion'] or '').strip()
                            score = attr_item['score']
                            dim_val = attr_item['dim_val']
                            dim_name = attr_item['dim_name']
                            
                            # 构建维度取值标注
                            dim_info = ""
                            if dim_val is not None and dim_name:
                                v_label = _group_value_label(dim_val)
                                dim_info = f"（{dim_name}: {v_label}）"
                            
                            if sug:
                                f.write(f"- **{attr_name}**{dim_info} (打分: {score}): {sug}\n")
                            else:
                                f.write(f"- **{attr_name}**{dim_info} (打分: {score}): （无建议）\n")
                        f.write("\n")
                    else:
                        f.write("> 无可用优化建议\n\n")

            # 属性优化优先顺序表
            f.write("#### 属性优化优先顺序表\n\n")
            prop_est = data.get('proportion_estimate')
            beta_plus_list = data.get('beta_plus', []) or []
            beta_minus_list = data.get('beta_minus', []) or []
            dims = data['segmentation_data'].get('segmentations', [])
            attr_map = data.get('attr_map', {})
            has_pos = data.get('dim_has_pos_value', []) or []
            has_neg = data.get('dim_has_neg_value', []) or []
            
            # 计算每个维度各取值的占比
            dim_proportions = {}  # dim_idx -> {0: prop, 1: prop, 2: prop}
            if isinstance(prop_est, list):
                for dim_idx in range(len(prop_est)):
                    if dim_idx < len(prop_est):
                        dim_prop_info = prop_est[dim_idx]
                        if isinstance(dim_prop_info, dict) and 'proportions' in dim_prop_info:
                            props = dim_prop_info['proportions']
                            if isinstance(props, list) and len(props) >= 3:
                                dim_proportions[dim_idx] = {
                                    0: float(props[0]) if props[0] is not None else 0.0,  # 负价值
                                    1: float(props[1]) if props[1] is not None else 0.0,  # 无价值
                                    2: float(props[2]) if props[2] is not None else 0.0   # 正价值
                                }
            
            # 收集所有属性的优化选项
            optimization_options = []
            
            # 遍历所有属性
            for attr_name, attr_info in attr_map.items():
                dim_idx = attr_info.get('dim_index')
                if dim_idx is None or dim_idx >= len(beta_plus_list):
                    continue
                
                beta_plus_val = beta_plus_list[dim_idx] if dim_idx < len(beta_plus_list) else 0.0
                beta_minus_val = beta_minus_list[dim_idx] if dim_idx < len(beta_minus_list) else 0.0
                dim_prop = dim_proportions.get(dim_idx, {0: 0.0, 1: 0.0, 2: 0.0})
                
                # 获取该属性的当前表现（从attr_desc_map中获取属性描述）
                attr_desc_map = data.get('attr_desc_map', {})
                attr_effect = attr_desc_map.get(attr_name, '（无描述）')
                
                # 获取属性方向（正向或反向）
                attr_direction = attr_info.get('direction', '正向')
                
                # 获取维度判定标准
                dim_info = dims[dim_idx] if dim_idx < len(dims) else None
                neg_criteria = dim_info.get('negative_value_criteria', '') if dim_info else ''
                neu_criteria = dim_info.get('neutral_value_criteria', '') if dim_info else ''
                pos_criteria = dim_info.get('positive_value_criteria', '') if dim_info else ''
                
                # 选项1：对负价值群体减少负面影响
                if dim_prop[0] > 0 and has_neg[dim_idx] if dim_idx < len(has_neg) else False:
                    # 重要性 = 负价值群体占比 * β^-值（使用绝对值用于排序）
                    importance = dim_prop[0] * abs(beta_minus_val)
                    
                    # 根据属性方向调整显示的优化方向
                    if attr_direction == '正向':
                        opt_direction = '对负价值群体减少负面影响'
                        opt_criteria = neg_criteria
                    else:
                        # 反向属性：显示为"对正价值群体提高满意度"
                        opt_direction = '对正价值群体提高满意度'
                        opt_criteria = pos_criteria
                    
                    optimization_options.append({
                        'attr_name': attr_name,
                        'direction': opt_direction,
                        'importance': importance,
                        'effect': attr_effect,
                        'criteria': opt_criteria,
                        'dim_idx': dim_idx,
                        'dim_val': 0,  # 负价值群体对应的维度取值（保持不变，用于查找匹配群体）
                        'beta_value': beta_minus_val,
                        'group_proportion': dim_prop[0]
                    })
                
                # 选项2：对无价值群体增强吸引力
                if dim_prop[1] > 0:
                    # 判断三个取值是否均不为空
                    has_all = (has_neg[dim_idx] if dim_idx < len(has_neg) else False) and \
                              (has_pos[dim_idx] if dim_idx < len(has_pos) else False)
                    
                    if has_all:
                        # 三个取值均不为空：使用平均值
                        beta_avg = (beta_plus_val + beta_minus_val) / 2.0
                        importance = dim_prop[1] * abs(beta_avg)
                        beta_used = beta_avg
                    else:
                        # 只有一个不为空：使用不为0的那个beta值
                        if has_neg[dim_idx] if dim_idx < len(has_neg) else False and abs(beta_minus_val) > 1e-12:
                            importance = dim_prop[1] * abs(beta_minus_val)
                            beta_used = beta_minus_val
                        elif has_pos[dim_idx] if dim_idx < len(has_pos) else False and abs(beta_plus_val) > 1e-12:
                            importance = dim_prop[1] * abs(beta_plus_val)
                            beta_used = beta_plus_val
                        else:
                            importance = 0.0
                            beta_used = 0.0
                    
                    if importance > 0:
                        optimization_options.append({
                            'attr_name': attr_name,
                            'direction': '对无价值群体增强吸引力',
                            'importance': importance,
                            'effect': attr_effect,
                            'criteria': neu_criteria,
                            'dim_idx': dim_idx,
                            'dim_val': 1,  # 无价值群体对应的维度取值
                            'beta_value': beta_used,  # 保存使用的beta值用于显示
                            'group_proportion': dim_prop[1]  # 保存群体占比
                        })
                
                # 选项3：对正价值群体提高满意度
                if dim_prop[2] > 0 and has_pos[dim_idx] if dim_idx < len(has_pos) else False:
                    # 重要性 = 正价值群体占比 * β^+值（使用绝对值用于排序）
                    importance = dim_prop[2] * abs(beta_plus_val)
                    
                    # 根据属性方向调整显示的优化方向
                    if attr_direction == '正向':
                        opt_direction = '对正价值群体提高满意度'
                        opt_criteria = pos_criteria
                    else:
                        # 反向属性：显示为"对负价值群体减少负面影响"
                        opt_direction = '对负价值群体减少负面影响'
                        opt_criteria = neg_criteria
                    
                    optimization_options.append({
                        'attr_name': attr_name,
                        'direction': opt_direction,
                        'importance': importance,
                        'effect': attr_effect,
                        'criteria': opt_criteria,
                        'dim_idx': dim_idx,
                        'dim_val': 2,  # 正价值群体对应的维度取值（保持不变，用于查找匹配群体）
                        'beta_value': beta_plus_val,
                        'group_proportion': dim_prop[2]
                    })
            
            # 按重要性降序排序
            optimization_options.sort(key=lambda x: x['importance'], reverse=True)
            
            # 生成表格
            if optimization_options:
                f.write("| 优先级 | 属性名称+优化方向 | 当前表现 | 用户群体判定标准 | β值 | 群体占比 | 重要性分数 | 参考优化建议 | 用户群体ID | 用户画像 | 用户打分 | 产品估值 |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for idx, opt in enumerate(optimization_options, 1):
                    attr_name = opt['attr_name']
                    direction = opt['direction']
                    effect = opt['effect'] or '（无描述）'
                    criteria = opt['criteria'] or '（无标准）'
                    importance = opt['importance']
                    beta_value = opt.get('beta_value', 0.0)
                    group_proportion = opt.get('group_proportion', 0.0)
                    dim_idx = opt['dim_idx']
                    dim_val = opt['dim_val']  # 0, 1, 或 2
                    
                    # 查找参考优化建议
                    reference_suggestion = "（无参考建议）"
                    reference_gid = "（无）"
                    reference_profile = "（无）"
                    reference_score = "（无）"
                    reference_valuation = "（无）"
                    
                    # 找到与该优化建议所属的维度和维度取值一致的用户群体
                    matching_groups = []
                    for _, row in sub_df.iterrows():
                        gid = str(row['uniq_id'])
                        group_dim_vals = _parse_group_dim_values_from_uniq_id(gid, num_dims)
                        if group_dim_vals is not None and dim_idx < len(group_dim_vals):
                            # 检查该群体在该维度上的取值是否匹配
                            if group_dim_vals[dim_idx] == dim_val:
                                # 获取该群体的估值（从当前行的psychological_price获取）
                                price_str = row.get('psychological_price', '0')
                                valuation = parse_price(price_str)
                                
                                # 获取该群体对该属性的打分
                                aa = safe_json_loads(row.get('attribute_analysis'))
                                attr_score = None
                                if isinstance(aa, list):
                                    for item in aa:
                                        if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                            attr_score = item.get('attribute_score')
                                            break
                                
                                if attr_score is not None:
                                    matching_groups.append({
                                        'gid': gid,
                                        'valuation': valuation,
                                        'attr_score': attr_score,
                                        'row': row
                                    })
                    
                    # 对匹配的群体进行排序
                    if matching_groups:
                        # 主标准：群体估值与最优定价的距离升序
                        # 副标准：群体对该属性的打分升序
                        optimal_price_val = float(data.get('optimal_price') or 0.0)
                        matching_groups.sort(key=lambda g: (
                            abs(g['valuation'] - optimal_price_val),  # 主标准：与最优定价的距离
                            g['attr_score']  # 副标准：属性打分升序
                        ))
                        
                        # 取排第一的群体对该属性的优化建议
                        top_group = matching_groups[0]
                        top_row = top_group['row']
                        reference_gid = top_group['gid']
                        reference_score = str(top_group['attr_score'])
                        reference_valuation = f"{top_group['valuation']:.2f}"
                        
                        # 获取用户画像
                        user_profile = top_row.get('user_profile', '')
                        if user_profile:
                            reference_profile = str(user_profile)
                        else:
                            reference_profile = "（无）"
                        
                        # 获取优化建议
                        top_aa = safe_json_loads(top_row.get('attribute_analysis'))
                        if isinstance(top_aa, list):
                            for item in top_aa:
                                if isinstance(item, dict) and item.get('attribute_name') == attr_name:
                                    ref_sug = item.get('attribute_optimization_suggestion', '')
                                    if ref_sug:
                                        reference_suggestion = ref_sug
                                    break
                    
                    # 限制文本长度，避免表格过宽
                    effect_short = effect[:80] + '...' if len(effect) > 80 else effect
                    criteria_short = criteria[:80] + '...' if len(criteria) > 80 else criteria
                    ref_sug_short = reference_suggestion[:100] + '...' if len(reference_suggestion) > 100 else reference_suggestion
                    profile_short = reference_profile[:100] + '...' if len(reference_profile) > 100 else reference_profile
                    
                    f.write(f"| {idx} | {attr_name} - {direction} | {effect_short} | {criteria_short} | {beta_value:.4f} | {group_proportion:.4f} | {importance:.6f} | {ref_sug_short} | {reference_gid} | {profile_short} | {reference_score} | {reference_valuation} |\n")
                f.write("\n")
            else:
                f.write("> 无可用优化选项\n\n")

            f.write("\n---\n\n")

    print(f"Analysis complete. Report generated at {output_file}")

if __name__ == "__main__":
    input_csv = 'dataset/processed/consumer_analysis.csv'
    output_md = 'dataset/processed/consumer_analysis.md'
    
    if os.path.exists(input_csv):
        analyze_consumer_data(input_csv, output_md)
    else:
        print(f"File {input_csv} not found.")