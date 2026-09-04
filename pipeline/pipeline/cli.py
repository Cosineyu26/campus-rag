import argparse
import logging

from .config import load_config
from .sync import run_full_sync


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pipeline", description="校园RAG数据管线")
    parser.add_argument("command", choices=["run"], default="run", nargs="?")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    report = run_full_sync(config)
    logging.info("同步完成: new=%s updated=%s stale=%s failed=%s",
                 report.new, report.updated, report.stale, report.failed)
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
