import argparse
import sys

from wsd.commands import commands


def print_wait(obj):
    print(obj)
    try:
        input("Press any key to continue, or CTRL+C to abort.")
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="print the config and parsed arguments"
    )

    subparsers = parser.add_subparsers(
        required=True, title="command", help="the command to execute"
    )

    for command in commands:
        command.register(subparsers)

    args = parser.parse_args()

    if args.debug:
        print_wait(args)

    args.action(args)
