import json
import os

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme(style="darkgrid")


def load_results(alg_name: str, env_key: str, run_number: int) -> pd.DataFrame:
    fp = os.path.join(
        "results/sacred", alg_name, env_key, str(run_number), "metrics.json"
    )
    with open(fp, "r") as f:
        raw_data = json.load(f)

    rets, ep_lengths = (
        raw_data["test_return"],
        raw_data["test_ep_length"],
    )

    assert len(rets["steps"]) == len(
        ep_lengths["steps"]
    ), "Test metrics array lengths do not match"

    data = []
    for i in range(len(rets["steps"])):
        _rets, _lengths = rets["values"][i], ep_lengths["values"][i]
        print(_rets, _lengths)
        assert len(_rets) == len(_lengths), "Test metrics array lengths do not match"

        data += [
            {
                "step": rets["steps"][i],
                "return": _rets[j],
                "ep_length": _lengths[j],
            }
            for j in range(len(_rets))
        ]

    return pd.DataFrame(data)


def plot_results(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    sns.lineplot(data=results, x="step", y="return", ax=ax[0])
    ax[0].set(xlabel="Step", ylabel="Return", title="Mean test return")

    sns.lineplot(data=results, x="step", y="ep_length", ax=ax[1])
    ax[1].set(xlabel="Step", ylabel="Episode length", title="Mean test episode length")

    plt.show()


if __name__ == "__main__":
    results = load_results("qmix", "lbforaging:Foraging-8x8-2p-3f-v2", 10)
    print(results.head())
    plot_results(results)
