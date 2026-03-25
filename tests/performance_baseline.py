import asyncio
import time
import statistics

# Simulated sync DB functions with some sleep to represent I/O
def sync_db_op(name, duration=0.1):
    # print(f"Starting sync operation: {name}")
    time.sleep(duration)
    # print(f"Finished sync operation: {name}")
    return []

async def heartbeat(interval=0.01):
    lags = []
    try:
        while True:
            start = time.perf_counter()
            await asyncio.sleep(interval)
            end = time.perf_counter()
            lag = end - start - interval
            if lag > 0:
                lags.append(lag)
    except asyncio.CancelledError:
        return lags

async def run_sync_benchmark(num_requests=5, op_duration=0.1):
    print(f"--- Running Sync Benchmark ({num_requests} requests, {op_duration}s each) ---")
    hb_task = asyncio.create_task(heartbeat())

    start_time = time.perf_counter()

    # Simulate concurrent requests in an async loop
    # In a real telegram bot, each handle_message is a separate task
    async def simulated_handler(i):
        # This mimics the current blocking behavior
        sync_db_op(f"req_{i}_get", op_duration)
        # ... some async work ...
        await asyncio.sleep(0.01)
        sync_db_op(f"req_{i}_set", op_duration)

    tasks = [asyncio.create_task(simulated_handler(i)) for i in range(num_requests)]
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    hb_task.cancel()
    lags = await hb_task

    total_time = end_time - start_time
    max_lag = max(lags) if lags else 0
    avg_lag = statistics.mean(lags) if lags else 0

    print(f"Total time: {total_time:.4f}s")
    print(f"Max event loop lag: {max_lag:.4f}s")
    print(f"Avg event loop lag: {avg_lag:.4f}s")
    return max_lag

async def run_async_benchmark(num_requests=5, op_duration=0.1):
    print(f"--- Running Async (Threaded) Benchmark ({num_requests} requests, {op_duration}s each) ---")
    hb_task = asyncio.create_task(heartbeat())

    start_time = time.perf_counter()

    async def simulated_handler(i):
        # This mimics the proposed optimized behavior
        await asyncio.to_thread(sync_db_op, f"req_{i}_get", op_duration)
        await asyncio.sleep(0.01)
        await asyncio.to_thread(sync_db_op, f"req_{i}_set", op_duration)

    tasks = [asyncio.create_task(simulated_handler(i)) for i in range(num_requests)]
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    hb_task.cancel()
    lags = await hb_task

    total_time = end_time - start_time
    max_lag = max(lags) if lags else 0
    avg_lag = statistics.mean(lags) if lags else 0

    print(f"Total time: {total_time:.4f}s")
    print(f"Max event loop lag: {max_lag:.4f}s")
    print(f"Avg event loop lag: {avg_lag:.4f}s")
    return max_lag

async def main():
    sync_max_lag = await run_sync_benchmark()
    print()
    async_max_lag = await run_async_benchmark()

    print("\n--- Summary ---")
    print(f"Sync Max Lag:  {sync_max_lag:.4f}s")
    print(f"Async Max Lag: {async_max_lag:.4f}s")
    improvement = (sync_max_lag - async_max_lag) / sync_max_lag * 100 if sync_max_lag > 0 else 0
    print(f"Lag Reduction: {improvement:.2f}%")

if __name__ == "__main__":
    asyncio.run(main())
