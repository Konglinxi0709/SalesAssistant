from typing import Any
import pandas as pd
import json
from processes.consumer_analysis import analyze_product_market


def run_optimization_analysis(input_file: str, output_file: str):
    """
    将optimized_consumer_feedback_data.csv处理为optimization_analysis_data.csv，添加优化后的市场分析指标
    
    参数:
    - input_file: 输入的CSV文件路径（optimized_consumer_feedback_data.csv）
    - output_file: 输出的CSV文件路径（optimization_analysis_data.csv）
    """
    # 读取输入文件
    df = pd.read_csv(input_file)
    print(f"成功读取数据，共 {len(df)} 行记录")
    
    # 检查必要的列是否存在
    required_columns = ['product_uniq_id', 'uniq_id', 'proportion_estimate', 'psychological_price', 'cost_estimate']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"输入文件缺少必要的列: {missing_columns}")
    
    # 定义需要重命名为_org的字段（consumer_analysis.py生成的所有分析指标字段）
    analysis_fields_to_rename = [
        'beta_0', 'beta_plus_coefficients', 'beta_minus_coefficients', 'dimension_variances',
        'importance_indicators', 'optimal_price', 'max_total_profit',
        'optimal_price_center_group_ids',
        'r_squared', 'mse', 'mae', 'rmse', 'k_attr_scores'
    ]
    
    # 定义用户级别字段（需要聚合到users_data）
    user_level_fields = ['uniq_id', 'attribute_analysis', 'psychological_price', 
                        'user_profile', 'user_profile_rationale', 'consumer_definition']
    
    # 按产品分组
    product_groups = df.groupby('product_uniq_id')
    print(f"发现 {len(product_groups)} 个不同产品")
    
    # 存储处理后的数据（一行一个产品）
    processed_rows = []
    analyzed_count = 0
    failed_count = 0
    
    # 遍历每个产品
    for product_id, product_data in product_groups:
        product_name = product_data.iloc[0].get('modified_name', product_id)
        print(f"正在分析产品: {product_name} (共 {len(product_data)} 个消费者)")
        
        # 取第一行作为产品级别数据
        product_row = product_data.iloc[0].copy()
        
        # 将product_uniq_id重命名为uniq_id
        product_row['uniq_id'] = product_row['product_uniq_id']
        
        # 1. 先将旧的分析字段重命名为_org形式
        for field in analysis_fields_to_rename:
            if field in product_row:
                old_value = product_row[field]
                new_field_name = f"{field}_org"
                product_row[new_field_name] = old_value
                # 删除原字段（稍后会用新计算的值填充）
                del product_row[field]
        
        # 2. 将原有的users_data字段重命名为users_data_org（如果存在）
        if 'users_data' in product_row:
            old_users_data = product_row['users_data']
            product_row['users_data_org'] = old_users_data
            # 不删除原字段，稍后会生成新的users_data
        
        # 3. 使用优化后的产品设计文档（hard_design, core_features, value_proposition）
        # 这些字段已经在优化过程中更新，直接使用即可
        
        # 4. 聚合用户数据（使用优化后的用户反馈）
        users_data = []
        for _, user_row in product_data.iterrows():
            user_dict = {}
            for field in user_level_fields:
                if field in user_row:
                    # 将uniq_id重命名为user_uniq_id
                    if field == 'uniq_id':
                        user_dict['user_uniq_id'] = user_row[field]
                    else:
                        user_dict[field] = user_row[field]
            users_data.append(user_dict)
        
        # 将新的users_data添加到产品行（保留users_data_org）
        product_row['users_data'] = json.dumps(users_data, ensure_ascii=False)
        
        # 删除product_uniq_id字段（已重命名为uniq_id）
        if 'product_uniq_id' in product_row:
            del product_row['product_uniq_id']
        
        # 5. 使用优化后的数据重新分析该产品
        analysis_result = analyze_product_market(product_row)
        
        if analysis_result:
            # 6. 用新计算的分析指标替换旧字段（已删除，现在直接添加新值）
            for key, value in analysis_result.items():
                product_row[key] = value
            processed_rows.append(product_row)
            analyzed_count += 1
        else:
            # 即使分析失败，也保留原始数据，但添加空指标字段（设为NaN）以保持输出结构一致
            print(f"警告: 产品 {product_name} 分析失败，保留原始数据但指标字段为空")
            for field in analysis_fields_to_rename:
                product_row[field] = None
            processed_rows.append(product_row)
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
    print(f"  总产品行数: {len(result_df)}")
    print(f"  结果已保存到: {output_file}")


if __name__ == "__main__":
    input_file = 'dataset/processed/optimized_consumer_feedback_data.csv'
    output_file = 'dataset/processed/optimization_analysis_data.csv'
    
    run_optimization_analysis(input_file, output_file)

