import torch

def continuous_batching_sim(arrival_times: torch.Tensor,
                            tokens_needed: torch.Tensor,
                            max_batch_size: int) -> dict:
    n = len(arrival_times)
    if n == 0:
        return {'total_time': 0, 'avg_latency': 0.0,
                'avg_ttft': 0.0, 'throughput': 0.0}

    # Keep correspondence; sort by arrival (stable)
    order = torch.argsort(arrival_times)
    arrival_times = arrival_times[order].tolist()
    tokens_needed = tokens_needed[order].tolist()

    t = 0
    queue = []            # indices currently being decoded
    waiting = []          # indices arrived but not yet admitted
    status = {i: 'not_arrived' for i in range(n)}
    tokens_left = {i: tokens_needed[i] for i in range(n)}
    all_tokens_left = sum(tokens_needed)
    completion_time = {i: 0 for i in range(n)}
    start_time = {i: 0 for i in range(n)}

    while all_tokens_left:
        # Admit from waiting first
        while len(queue) < max_batch_size and waiting:
            i = waiting.pop(0)
            status[i] = 'working'
            queue.append(i)
            start_time[i] = t

        # Admit new arrivals (in arrival order)
        for i in range(n):
            if status[i] != 'not_arrived':
                continue
            if arrival_times[i] <= t:
                if len(queue) < max_batch_size:
                    queue.append(i)
                    status[i] = 'working'
                    start_time[i] = t
                else:
                    waiting.append(i)
                    status[i] = 'waiting'
            else:
                break  # sorted, so later ones haven't arrived

        # Decode one token per active request
        finished = []
        for r in queue:
            tokens_left[r] -= 1
            all_tokens_left -= 1
            if tokens_left[r] == 0:
                completion_time[r] = t
                finished.append(r)
        for r in finished:
            queue.remove(r)
            status[r] = 'done'

        t += 1

    return {
        'total_time': t,
        'avg_latency': round(
            sum(completion_time[j] - arrival_times[j] for j in range(n)) / n, 4),
        'avg_ttft': round(
            sum(start_time[i] - arrival_times[i] for i in range(n)) / n, 4),
        'throughput': round(sum(tokens_needed) / t, 4) if t else 0.0,
    }