import gc

import torch

from weightd import list_shared_weights, load_shared_weights


def main() -> None:
    print("Available weight bundles:")
    for name in list_shared_weights():
        print(f"  - {name}")

    if not list_shared_weights():
        print("No weight bundles available. Please run the server first.")
        return

    name = list_shared_weights()[0]
    weights = load_shared_weights(name)

    print(f"\nLoaded weight bundle: {weights.name}")
    weight = weights["layer.weight"]
    bias = weights["layer.bias"]

    x = torch.ones(4, dtype=torch.float32, device=weight.device)

    output = weight @ x + bias
    print(f"Output: {output}")

    # Clean up
    del weights, weight, bias, x, output
    gc.collect()


if __name__ == "__main__":
    main()
