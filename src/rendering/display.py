import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML
from datetime import datetime

def get_progress_fn():
    times = []
    x_data = []
    y_data = []
    ydataerr = []

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        current_time = times[-1]-times[0]
        num_steps = x_data[-1]
        latest_reward = y_data[-1]
        latest_reward_std = ydataerr[-1]

        max_reward = max(y_data)

        if not 'training/sps' in metrics.keys():
            metrics['training/sps'] = 0

        report = f"""
            Time: {current_time}

            Total Steps Taken: {num_steps}

            Training Steps Per Second: {metrics['training/sps']:0.2f}

            Evaluation Metrics:
            - Episode Rewards:
            - Latest Reward: {latest_reward:0.2f}
            - Standard Deviation: {latest_reward_std:0.2f}
            - Maximum Recorded Reward: {max_reward:0.2f}

            --------------------------------------------
            """

        print(report)
    
    return progress
  