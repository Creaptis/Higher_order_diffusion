def to_str(x):
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return f"{x:.6f}"
    if isinstance(x, int):
        return f"{x}"
    if isinstance(x, str):
        return x
    else:
        return x


def print_metrics(metrics_dict):
    status_msg = " - ".join([f"{k}: {to_str(v)}" for k, v in metrics_dict.items()])
    print(status_msg)
