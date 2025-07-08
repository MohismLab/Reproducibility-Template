import logging

# set the logger of the whole experiment
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    logger.info("Training is now starting")


if __name__ == "__main__":
    main()
