import traceback
import json
import httpx
from openai import APIConnectionError, InternalServerError
from tenacity import (
    Retrying,
    retry_if_exception_type,
    wait_random_exponential,
    stop_after_attempt
)
from typing import Awaitable, Any, Callable, Dict, List
import pandas as pd
import asyncio
import os
from tqdm.asyncio import tqdm

async def async_call_with_retry(func: Awaitable, logger, max_retries, *args, **kwargs) -> Any:
    """带重试机制的异步调用函数"""
    try:
        for attempt in Retrying(
            wait=wait_random_exponential(min=1, max=60),
            stop=stop_after_attempt(max_retries + 1),
            reraise=True,
            retry=(
                retry_if_exception_type(httpx.ConnectTimeout) |
                retry_if_exception_type(httpx.ReadTimeout) |
                retry_if_exception_type(httpx.NetworkError) |
                retry_if_exception_type(ValueError) |
                retry_if_exception_type(APIConnectionError) | # Added for openai connection errors
                retry_if_exception_type(InternalServerError) # Added for openai internal server errors
            )
        ):
            with attempt:
                try:
                    return await func(*args, **kwargs)
                except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError, ValueError, APIConnectionError, InternalServerError) as e:
                    logger(f"[warrning]: LLM error during async call: {str(e)} - Attempt {attempt.retry_state.attempt_number}")
                    raise
                except Exception as e:
                    logger(f"[error]: Unexpected error during async call: {str(e)}")
                    raise
    except Exception as e:
        logger(f"[error]: Failed after {max_retries} retries: {str(e)}")
        raise


async def run_concurrent_tasks(
    input_csv_path: str,
    output_csv_path: str,
    log_folder_path: str,
    task_callback: Callable[[Dict[str, Any], Callable[[str], None]], Awaitable[Dict[str, Any]]],
    filter_callback: Callable[[Dict[str, Any], pd.DataFrame], bool],
    concurrent_tasks_limit: int = 10,
    max_retries: int = 3,
    max_total_tasks: int = None  # 新增参数，可选，控制最大执行任务总数
) -> None:
    """运行和管理异步并发多任务的函数。

    参数说明:
        input_csv_path: 输入CSV路径
        output_csv_path: 输出CSV路径
        log_folder_path: 日志文件夹
        task_callback: 任务处理回调
        filter_callback: 过滤回调
        concurrent_tasks_limit: 并发任务数
        max_retries: 最大重试次数
        max_total_tasks: 仅处理最多该数量的任务（可选，None表示不限制）
    """

    # 1. 读取输入csv文件
    input_df = pd.read_csv(input_csv_path)
    all_rows = input_df.to_dict(orient='records')

    # 应用最大任务数限制（如果设置了max_total_tasks）
    if max_total_tasks is not None and max_total_tasks > 0:
        all_rows = all_rows[:max_total_tasks]

    # 读取现有输出文件以便进行过滤
    existing_output_df = pd.DataFrame()
    if os.path.exists(output_csv_path):
        existing_output_df = pd.read_csv(output_csv_path)

    # 2. 用第二个回调函数对所有行进行判断，只保留需要处理的行
    rows_to_process = [row for row in all_rows if filter_callback(row, existing_output_df)]

    total_input_rows = len(all_rows)
    num_rows_to_process = len(rows_to_process)

    if num_rows_to_process == 0:
        print("没有需要处理的任务。")
        return

    # 3. 按每`并发任务个数`个行为一组，对所有行进行分组
    task_groups = [rows_to_process[i:i + concurrent_tasks_limit] for i in range(0, num_rows_to_process, concurrent_tasks_limit)]

    # 4. 定义一个日志字典，键为每个行的uniq_id，值为一个日志数组，以及一个结果状态字典，键为uniq_id，值为布尔值初始化为False，再初始化一个tqdm进度条，总个数为分组后任务的组数
    logs: Dict[str, List[str]] = {row['uniq_id']: [] for row in rows_to_process}
    results_status: Dict[str, bool] = {row['uniq_id']: False for row in rows_to_process}
    output_rows: List[Dict[str, Any]] = []

    def build_task_logger(uniq_id):
        def task_logger(message: str):
            logs[uniq_id].append(message)
        return task_logger

    os.makedirs(log_folder_path, exist_ok=True)

    with tqdm(total=len(task_groups), desc="处理任务组") as pbar:
        for group_idx, group in enumerate(task_groups):
            tasks = []
            for row in group:
                uniq_id = row['uniq_id']
                task_logger = build_task_logger(uniq_id)
                async def wrapped_task(row=row, uniq_id=uniq_id, task_logger=task_logger):
                    try:
                        # 将输入row复制一份，避免在任务中修改原始数据
                        processed_row = await async_call_with_retry(task_callback, task_logger, max_retries, row, task_logger)
                        # 确保uniq_id在结果中，即使回调函数没有返回
                        if 'uniq_id' not in processed_row:
                            processed_row['uniq_id'] = uniq_id
                        results_status[uniq_id] = True
                        return processed_row
                    except Exception as e:
                        task_logger(f"任务最终失败: {str(e)}")
                        results_status[uniq_id] = False
                        # 返回一个空值字典，包含uniq_id以便后续处理
                        return {'uniq_id': uniq_id}

                tasks.append(wrapped_task())

            # 2. 并发的执行该组的所有封装后的任务函数
            group_results = await asyncio.gather(*tasks)
            # 这里仅添加成功的结果
            output_rows.extend([result for result in group_results if results_status.get(result.get('uniq_id'))])

            # 3. 待所有任务函数执行完毕后，将所有任务执行的结果写入输出csv文件；
            # 确保所有列都存在，否则会报错
            if output_rows:
                # 获取所有可能的列名，包括原始DataFrame的列和新生成的列
                all_columns = list(input_df.columns.values) + list(output_rows[0].keys())
                all_columns = sorted(list(set(all_columns))) # 去重并排序

                temp_output_df = pd.DataFrame(output_rows, columns=all_columns)
                temp_output_df.to_csv(output_csv_path, index=False)

            # 并且将日志字典写入到日志文件夹中，每个任务分别写入到`日志文件夹/任务名.log`文件中，如果原本已有该文件则覆盖；
            for row in group:
                uniq_id = row['uniq_id']
                log_file_path = os.path.join(log_folder_path, f"{uniq_id}.log")
                with open(log_file_path, 'w', encoding='utf-8') as f:
                    for log_entry in logs[uniq_id]:
                        f.write(log_entry + '\n')

            # 统计当前组的成功和失败任务
            group_successful = sum(1 for row in group if results_status.get(row['uniq_id'], False))
            group_failed = len(group) - group_successful
            group_failed_uniq_ids = [row['uniq_id'] for row in group if not results_status.get(row['uniq_id'], False)]

            print(f"\n第 {group_idx + 1} 组任务完成:")
            print(f"  成功任务数: {group_successful}")
            print(f"  失败任务数: {group_failed}")
            if group_failed > 0:
                print(f"  失败任务uniq_id: {group_failed_uniq_ids}")

            pbar.update(1)

    # 统计结果
    successful_tasks = sum(results_status.values())
    failed_tasks = num_rows_to_process - successful_tasks
    failed_uniq_ids = [uniq_id for uniq_id, status in results_status.items() if not status]

    print(f"\n总任务个数 (输入行数): {total_input_rows}")
    print(f"需要执行的任务个数: {num_rows_to_process}")
    print(f"成功执行的任务个数: {successful_tasks}")
    print(f"失败的任务个数: {failed_tasks}")

    if failed_tasks > 0:
        print("失败任务的uniq_id (前20个):")
        for i, uniq_id in enumerate(failed_uniq_ids):
            if i >= 20:
                print("...")
                break
            print(f"- {uniq_id}")
