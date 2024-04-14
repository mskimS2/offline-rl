import argparse


def get_config():
    parser = argparse.ArgumentParser(description="PPO")

    return parser.parse_args()
