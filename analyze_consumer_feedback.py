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
    plt.ylabel('Beta Coefficient')
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
        beta_coeffs = safe_json_loads(row['beta_coefficients'])
        segmentation = safe_json_loads(row['segmentation_result'])
        attr_analysis = safe_json_loads(row['attribute_analysis'])
        k_attr_scores = safe_json_loads(row.get('k_attr_scores'))
        uniq_id = str(row['uniq_id'])
        price_str = row.get('retail_price', '1') # 获取价格

        if not product_name or beta_coeffs is None or segmentation is None or attr_analysis is None:
            continue

        # 初始化产品数据结构
        if product_name not in products_data:
            neg_beta_count = sum(1 for b in beta_coeffs if b < 0)
            price = parse_price(price_str)
            norm_betas = [b / price for b in beta_coeffs] # 计算归一化Beta
            
            # 构建属性到维度的映射表
            attr_map = {}
            dims = segmentation.get('segmentations', [])
            
            # 提取产品所有属性的描述备用
            attr_desc_map = {}
            for item in segmentation.get('total_attributes', []):
                attr_desc_map[item['attribute_name']] = item['attribute_description']

            for idx, dim in enumerate(dims):
                dim_name = dim.get('dimension_name', f"Dimension {idx+1}")
                beta_val = beta_coeffs[idx] if idx < len(beta_coeffs) else 0
                norm_beta_val = norm_betas[idx] if idx < len(norm_betas) else 0
                
                # 正向属性
                for attr in dim.get('positive_attribute_names', []):
                    attr_map[attr] = {
                        'dim_index': idx,
                        'dim_name': dim_name,
                        'beta': beta_val,
                        'norm_beta': norm_beta_val,
                        'direction': '正向'
                    }
                # 反向属性
                for attr in dim.get('negative_attribute_names', []):
                    attr_map[attr] = {
                        'dim_index': idx,
                        'dim_name': dim_name,
                        'beta': beta_val,
                        'norm_beta': norm_beta_val,
                        'direction': '反向'
                    }

            products_data[product_name] = {
                'product_uniq_id': product_uniq_id,
                'betas': beta_coeffs,
                'norm_betas': norm_betas,
                'price': price,
                'neg_beta_count': neg_beta_count,
                'attr_map': attr_map,
                'segmentation_data': segmentation,
                'attr_desc_map': attr_desc_map,
                'stats': {},
                'k_scores': k_attr_scores if isinstance(k_attr_scores, dict) else {}
            }

        num_dims = len(beta_coeffs)
        parts = uniq_id.split('_')
        
        # 提取用户维度取值
        if len(parts) >= num_dims:
            try:
                user_dim_values = [int(x) for x in parts[-num_dims:]]
            except ValueError:
                continue
        else:
            continue

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

    # k值为正比例 -> beta正/负计数
    k_ratio_buckets = {}  # ratio_str -> {'beta_pos': cnt, 'beta_neg': cnt}
    
    # 绘图数据 (normalized_beta, average_score)
    plot_points = {
        'neg': [], # 负价值图表
        'neu': [], # 无价值图表
        'pos': []  # 正价值图表
    }

    for p_name, p_data in products_data.items():
        for attr_name, stats in p_data['stats'].items():
            info = p_data['attr_map'][attr_name]
            norm_beta = info['beta'] # info['norm_beta'] # 使用归一化Beta
            direction = info['direction']
            k_scores_map = p_data.get('k_scores', {})
            
            # 汇总全局数据（仍然保持 accuracy 统计用于文本报告）
            for i in range(3):
                global_stats[i]['correct'] += stats[i]['correct']
                global_stats[i]['total'] += stats[i]['total']
            
            # 准备绘图数据：修改为 (Normalized Beta, Average Score)
            
            # 负价值图表 (Neg Value Chart)
            # 正向相关 -> 取Dim取值0
            # 反向相关 -> 取Dim取值2
            target_idx = 0 if direction == '正向' else 2
            if stats[target_idx]['total'] > 0:
                avg_score = stats[target_idx]['score_sum'] / stats[target_idx]['total']
                plot_points['neg'].append((norm_beta, avg_score))

            # 无价值图表 (Neutral Value Chart)
            # 始终取Dim取值1
            if stats[1]['total'] > 0:
                avg_score = stats[1]['score_sum'] / stats[1]['total']
                plot_points['neu'].append((norm_beta, avg_score))

            # 正价值图表 (Pos Value Chart)
            # 正向相关 -> 取Dim取值2
            # 反向相关 -> 取Dim取值0
            target_idx = 2 if direction == '正向' else 0
            if stats[target_idx]['total'] > 0:
                avg_score = stats[target_idx]['score_sum'] / stats[target_idx]['total']
                plot_points['pos'].append((norm_beta, avg_score))

        # 计算维度层面的“k为正”比例分布
        betas = p_data['betas']
        k_scores_map = p_data.get('k_scores', {})
        dims = p_data['segmentation_data'].get('segmentations', [])
        for idx, dim in enumerate(dims):
            attrs = dim.get('positive_attribute_names', []) + dim.get('negative_attribute_names', [])
            valid_attrs = [a for a in attrs if a in k_scores_map]
            if not valid_attrs:
                continue
            pos_k_cnt = sum(1 for a in valid_attrs if k_scores_map[a] > 0)
            ratio = pos_k_cnt / len(valid_attrs)
            ratio_key = f"{ratio:.2f}"
            beta_val = betas[idx] if idx < len(betas) else 0
            bucket = k_ratio_buckets.setdefault(ratio_key, {'beta_pos': 0, 'beta_neg': 0})
            if beta_val > 0:
                bucket['beta_pos'] += 1
            elif beta_val < 0:
                bucket['beta_neg'] += 1

    # 生成图表：散点图 (修改了标题和文件名以体现改动)
    plot_scatter(plot_points['neg'], 'Negative Value Group Avg Score vs Beta', os.path.join(img_dir, 'neg_value_avg_score.png'), 'red')
    plot_scatter(plot_points['neu'], 'Neutral Value Group Avg Score vs Beta', os.path.join(img_dir, 'neu_value_avg_score.png'), 'blue')
    plot_scatter(plot_points['pos'], 'Positive Value Group Avg Score vs Beta', os.path.join(img_dir, 'pos_value_avg_score.png'), 'green')

    # 生成k比例与β方向分布的水平堆叠图
    def plot_k_ratio_bucket(bucket_stats, filename):
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
        ax.set_title('各维度k>0比例分组的β正/负占比（柱高=属性个数占比）')
        
        # 调整x轴范围，为外部标注留出空间
        max_width = max(neg_vals[i] + pos_vals[i] for i in range(len(y_labels)))
        ax.set_xlim(0, max(100, max_width + 15))  # 留出15%的空间用于标注
        
        # 设置y轴范围
        ax.set_ylim(-spacing, current_y)
        
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    k_ratio_chart = os.path.join(img_dir, 'k_ratio_beta_distribution.png')
    plot_k_ratio_bucket(k_ratio_buckets, k_ratio_chart)

    # 计算总准确率
    total_correct_all = sum(g['correct'] for g in global_stats.values())
    total_count_all = sum(g['total'] for g in global_stats.values())

    # 计算总体β为正占比（所有产品、所有维度）
    total_beta_count = 0
    pos_beta_count = 0
    for p_data in products_data.values():
        betas = p_data.get('betas', [])
        for b in betas:
            if isinstance(b, (int, float)):
                total_beta_count += 1
                if b > 0:
                    pos_beta_count += 1
    
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

        # k比例与β方向分布图
        if total_beta_count > 0:
            f.write(f"- **总体β为正占比**: {pos_beta_count}/{total_beta_count} ({pos_beta_count/total_beta_count:.2%})\n\n")
        else:
            f.write("- **总体β为正占比**: 无有效β数据\n\n")

        f.write("### k为正比例分布下的β方向占比\n")
        if k_ratio_buckets:
            f.write("![k_ratio_beta_distribution](pictures/k_ratio_beta_distribution.png)\n\n")
        else:
            f.write("> 无可用数据生成该图\n\n")
        
        f.write("## 2. 平均分与Beta散点图 (Avg Score vs Beta)\n\n")
        f.write("> **注**: Beta = Beta Coefficient\n\n")
        f.write("### 负价值群体平均分 vs Beta\n")
        f.write("![Negative Value Group Avg Score](pictures/neg_value_avg_score.png)\n\n")
        f.write("### 无价值群体平均分 vs Beta\n")
        f.write("![Neutral Value Group Avg Score](pictures/neu_value_avg_score.png)\n\n")
        f.write("### 正价值群体平均分 vs Beta\n")
        f.write("![Positive Value Group Avg Score](pictures/pos_value_avg_score.png)\n\n")

        f.write("## 3. 产品详细分析 (Product Details)\n\n")
        
        for product_name, data in sorted_products:
            f.write(f"### Product: {product_name}\n")
            f.write(f"**ID**: {data['product_uniq_id']}\n")
            f.write(f"**Price**: {data['price']}\n\n")
            f.write(f"**Beta Coefficients**: {data['betas']}\n")
            f.write(f"**Normalized Betas**: {[round(b, 6) for b in data['norm_betas']]}\n\n")
            
            # 属性统计表格
            f.write("| 属性名称 | 对应维度 | 对应维度的β值 | 对应方向 | 负价值群体准确率 | 无价值群体准确率 | 正价值群体准确率 | 总准确率 | 分数k值 |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
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
                
                # 获取k值并格式化
                k_value = k_scores.get(attr_name)
                if k_value is not None:
                    k_str = f"{k_value:.4f}"
                    # 如果k值为负，用双等号高亮
                    if k_value < 0:
                        k_str = f"=={k_str}=="
                else:
                    k_str = "-"
                
                f.write(f"| {attr_name} | {info['dim_name']} | {info['beta']} | {info['direction']} | {neg_acc_str} | {neu_acc_str} | {pos_acc_str} | {total_acc_str} | {k_str} |\n")
            
            # 添加潜在用户群体定义
            user_group_def = data['segmentation_data'].get('user_group', 'N/A')
            f.write(f"\n**潜在用户群体定义**: \n{user_group_def}\n\n")

            # 处理负Beta维度的详细信息
            if data['neg_beta_count'] > 0:
                f.write("#### 负Beta维度详细分析\n\n")
                dims = data['segmentation_data'].get('segmentations', [])
                
                for idx, beta in enumerate(data['betas']):
                    if beta < 0 and idx < len(dims):
                        dim_info = dims[idx]
                        dim_name = dim_info.get('dimension_name', f"Dimension {idx+1}")
                        
                        f.write(f"**维度**: {dim_name} (Beta: {beta}, Norm Beta: {data['norm_betas'][idx]:.6f})\n\n")
                        
                        # 列出相关属性及其实际表现
                        pos_attrs = dim_info.get('positive_attribute_names', [])
                        neg_attrs = dim_info.get('negative_attribute_names', [])
                        all_related_attrs = pos_attrs + neg_attrs
                        
                        if all_related_attrs:
                            f.write("- **相关属性及用户打分分布**:\n")
                            for attr in all_related_attrs:
                                direction = "正向" if attr in pos_attrs else "反向"
                                desc = data['attr_desc_map'].get(attr, "无描述")
                                k_val = data.get('k_scores', {}).get(attr)
                                if isinstance(k_val, (int, float)):
                                    if k_val < 0:
                                        k_str = f", ==k={k_val:.4f}=="
                                    else:
                                        k_str = f", k={k_val:.4f}"
                                else:
                                    k_str = ""
                                f.write(f"  - **{attr}** ({direction}{k_str}): {desc}\n")
                                
                                # === 新增：绘制该属性的堆叠条形图 ===
                                if attr in data['stats']:
                                    attr_stats = data['stats'][attr]
                                    # 提取各组的分布
                                    distribution_data = {
                                        '负价值': attr_stats[0]['score_counts'],
                                        '无价值': attr_stats[1]['score_counts'],
                                        '正价值': attr_stats[2]['score_counts']
                                    }
                                    
                                    # 清理文件名
                                    safe_attr_name = re.sub(r'\W+', '_', attr)
                                    chart_filename = f"dist_{data['product_uniq_id']}_d{idx}_{safe_attr_name}.png"
                                    chart_path = os.path.join(img_dir, chart_filename)
                                    
                                    plot_stacked_bar(
                                        distribution_data, 
                                        f"Score Distribution for '{attr}'", 
                                        chart_path
                                    )
                                    
                                    f.write(f"\n    ![Score Distribution for {attr}](pictures/{chart_filename})\n\n")
                                # ==================================
                        
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

            f.write("\n---\n\n")

    print(f"Analysis complete. Report generated at {output_file}")

if __name__ == "__main__":
    input_csv = 'dataset/processed/consumer_analysis.csv'
    output_md = 'dataset/processed/consumer_analysis.md'
    
    if os.path.exists(input_csv):
        analyze_consumer_data(input_csv, output_md)
    else:
        print(f"File {input_csv} not found.")