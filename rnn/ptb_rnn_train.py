import logging as log

import sys
sys.path.append("..")

from lib import ptb, vocab

BATCH = 64
NUM_STEPS = 10

if __name__ == "__main__":
    log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("Loading PTB dataset...")
    data = ptb.PTBDataset("data", vocab.Vocab(), num_steps=10)
    log.info("Loaded: %s", data)

    for train_x, train_y in data.iterate_train(BATCH):
        print(len(train_x), len(train_y))
    pass
