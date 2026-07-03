"""Allow running RLtune as a package: python -m RLtune [--pytorch]

By default runs JAX training. Pass --pytorch to run PyTorch training.
"""

import sys


def main():
    if "--pytorch" in sys.argv:
        sys.argv.remove("--pytorch")
        from RLtune.train_pytorch import main as train_main
        train_main()
    else:
        from RLtune.train import main as train_main
        train_main()


if __name__ == "__main__":
    main()
