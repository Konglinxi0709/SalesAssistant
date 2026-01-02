from utils.multi_task import run_concurrent_tasks
from processes.consumer_feedback import consumer_feedback_task, filter_existing_feedback
from processes.market_segment import market_segment_task, filter_existing_segmentation
from processes.proportion_estimate import proportion_estimate_task, filter_existing_proportion
from processes.product_optimization import product_optimization_task, filter_existing_optimization
from processes.base_product_optimization import base_product_optimization_task, filter_existing_base_optimization
from processes.consumer_preprocess import run_consumer_preprocess
from processes.optimization_analysis import run_optimization_analysis

import asyncio
async def main():
    #await run_concurrent_tasks(
    #    'dataset/processed/with_category_only_2C_data.csv', 
    #    'dataset/processed/with_segment_data.csv', 
    #    'logs/market_segment_logs', 
    #    market_segment_task, 
    #    filter_existing_segmentation,
    #    concurrent_tasks_limit=50,
    #    max_retries=2)
    
    #await run_concurrent_tasks(
    #    'dataset/processed/with_segment_data.csv', 
    #    'dataset/processed/with_proportion_data.csv', 
    #    'logs/proportion_estimate_logs', 
    #    proportion_estimate_task, 
    #    filter_existing_proportion,
    #    concurrent_tasks_limit=50,
    #    max_retries=2)

    #run_consumer_preprocess('dataset/processed/with_proportion_data.csv', 'dataset/processed/consumer_data.csv', max_total_product=50)
    
    #await run_concurrent_tasks(
    #    'dataset/processed/consumer_data.csv',
    #    'dataset/processed/consumer_feedback_data.csv',
    #    'logs/consumer_feedback_logs',
    #    consumer_feedback_task,
    #    filter_existing_feedback,
    #    concurrent_tasks_limit=500,
    #    max_retries=2
    #)
    
    # 基线产品优化
    
    await run_concurrent_tasks(
        'dataset/processed/with_consumer_analysis_data.csv',
        'dataset/processed/base_with_optimization_data.csv',
        'logs/base_product_optimization_logs',
        base_product_optimization_task,
        filter_existing_base_optimization,
        concurrent_tasks_limit=500,
        max_retries=2,
        max_total_tasks=50,
    )

    run_consumer_preprocess('dataset/processed/base_with_optimization_data.csv', 'dataset/processed/base_optimized_consumer_data.csv')

    await run_concurrent_tasks(
        'dataset/processed/base_optimized_consumer_data.csv',
        'dataset/processed/base_optimized_consumer_feedback_data.csv',
        'logs/base_optimized_consumer_feedback_logs',
        consumer_feedback_task,
        filter_existing_feedback,
        concurrent_tasks_limit=500,
        max_retries=2
    )
    run_optimization_analysis(
        'dataset/processed/base_optimized_consumer_feedback_data.csv',
        'dataset/processed/base_optimization_analysis_data.csv'
    )
    # 控制台运行: nohup python main.py > nohup.log 2>&1 &；disown
    # 控制台运行: python analyze_optimization.py --input dataset/processed/base_optimization_analysis_data.csv --output dataset/processed/base_optimization_analysis.md


    # 反馈产品优化
    #await run_concurrent_tasks(
    #    'dataset/processed/with_consumer_analysis_data.csv', 
    #    'dataset/processed/with_optimization_data.csv', 
    #    'logs/product_optimization_logs', 
    #    product_optimization_task, 
    #    filter_existing_optimization,
    #    concurrent_tasks_limit=50,
    #    max_retries=2,
    #    max_total_tasks=500)
#
    #run_consumer_preprocess('dataset/processed/with_optimization_data.csv', 'dataset/processed/optimized_consumer_data.csv')
#
    #await run_concurrent_tasks(
    #    'dataset/processed/optimized_consumer_data.csv',
    #    'dataset/processed/optimized_consumer_feedback_data.csv',
    #    'logs/optimized_consumer_feedback_logs',
    #    consumer_feedback_task,
    #    filter_existing_feedback,
    #    concurrent_tasks_limit=500,
    #    max_retries=2)

    # 反馈产品优化（加上无价值群体）
    #await run_concurrent_tasks(
    #    'dataset/processed/with_consumer_analysis_data.csv', 
    #    'dataset/processed/with_optimization_data（加上无价值群体）.csv', 
    #    'logs/product_optimization_logs（加上无价值群体）', 
    #    product_optimization_task, 
    #    filter_existing_optimization,
    #    concurrent_tasks_limit=50,
    #    max_retries=2,
    #    max_total_tasks=500)
#
    #run_consumer_preprocess('dataset/processed/with_optimization_data（加上无价值群体）.csv', 'dataset/processed/optimized_consumer_data（加上无价值群体）.csv')

    #await run_concurrent_tasks(
    #    'dataset/processed/optimized_consumer_data（加上无价值群体）.csv',
    #    'dataset/processed/optimized_consumer_feedback_data（加上无价值群体）.csv',
    #    'logs/optimized_consumer_feedback_logs（加上无价值群体）',
    #    consumer_feedback_task,
    #    filter_existing_feedback,
    #    concurrent_tasks_limit=500,
    #    max_retries=2)

if __name__ == "__main__":
    asyncio.run(main())