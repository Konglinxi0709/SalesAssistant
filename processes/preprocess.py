import pandas as pd
import matplotlib.pyplot as plt

def run_preprocess(input_file, output_file):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 提取根节点类别
    def extract_root(category_tree):
        # 移除字符串两端的方括号和引号，然后按" >> "分割并取第一部分
        if isinstance(category_tree, str):
            clean_str = category_tree.strip('[]"')
            parts = clean_str.split(' >> ')
            if parts:
                return parts[0]
        return None

    df['root_category'] = df['product_category_tree'].apply(extract_root)

    # 对每个root_category分组，随机抽取一行
    sampled_df = df.groupby('root_category', as_index=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)
    
    # 对每一行的retail_price和discounted_price进行处理，乘以0.082并格式化为"{价格}￥"
    def format_price(price):
        try:
            price_num = float(price)
            new_price = price_num * 0.082
            return f"{new_price:.2f}￥"
        except:
            return ""

    sampled_df['retail_price'] = sampled_df['retail_price'].apply(format_price)
    sampled_df['discounted_price'] = sampled_df['discounted_price'].apply(format_price)
    print(sampled_df.head())

    # 将新表保存到输出文件路径
    sampled_df.to_csv(output_file, index=False)
