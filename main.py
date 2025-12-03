from utils.multi_task import run_concurrent_tasks
from processes.consumer_feedback import consumer_feedback_task, filter_existing_feedback
from processes.market_segment import market_segment_task, filter_existing_segmentation
from processes.proportion_estimate import proportion_estimate_task, filter_existing_proportion
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
    #
    #await run_concurrent_tasks(
    #    'dataset/processed/with_segment_data.csv', 
    #    'dataset/processed/with_proportion_data.csv', 
    #    'logs/proportion_estimate_logs', 
    #    proportion_estimate_task, 
    #    filter_existing_proportion,
    #    concurrent_tasks_limit=50,
    #    max_retries=2)

    await run_concurrent_tasks(
        'dataset/processed/consumer_data.csv',
        'dataset/processed/consumer_feedback_data.csv',
        'logs/consumer_feedback_logs',
        consumer_feedback_task,
        filter_existing_feedback,
        concurrent_tasks_limit=50,
        max_retries=2,
        max_total_tasks=200
    )

if __name__ == "__main__":
    asyncio.run(main())