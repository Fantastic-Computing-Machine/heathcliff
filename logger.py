from logging import getLogger, StreamHandler, Formatter, DEBUG, FileHandler

logger = getLogger(__name__)

handler = StreamHandler()

# write to file also

handler = FileHandler("log.log")
handler.setLevel(DEBUG)


formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)

logger.setLevel(DEBUG)
