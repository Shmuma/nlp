#!/usr/bin/env python
import json
from tqdm import tqdm
import argparse
import logging as log
import numpy as np
from smart_open import smart_open

from lib import data
from lib import model

TEST_FILE = "data/test.tsv.gz"


if __name__ == "__main__":
    log.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runfile", required=True, help="Name of runfile to use")
    parser.add_argument("-m", "--models", required=True, help="Json file with models to use")
    parser.add_argument("-o", "--output", required=True, help="Name of file to produce")
    args = parser.parse_args()

    runfile = data.read_runfile(args.runfile)
    log.info("Runfile: %s", json.dumps(runfile, indent='\t', sort_keys=True))

    log.info("Reading embeddings, it can take time")
    words, embeddings = data.read_embeddings(runfile['embeddings'])
    emb_len = embeddings.shape[1]
    log.info("We've read %d words from embeddings, it's size = %d", len(words), emb_len)

    mod_paths = data.read_runfile(args.models)
    models = {}

    for mod, path in mod_paths.items():
        m = model.create_model(1, emb_len, model.OUTPUTS[mod])
        m.load_weights(path)
        models[mod] = m

    samples = data.read_samples(TEST_FILE, words)
    text_tokens = data.tokenize_texts(samples['description'], words)

    with smart_open(args.output, "wt", encoding='utf-8') as fd:
        fd.write('tags\n')

        for tokens in tqdm(text_tokens):
            tags = []
            for name, mod in models.items():
                last_pred = None
                for win in data.iterate_batch_windows([tokens], model.WINDOW_SIZE):
                    batch_x = [embeddings[x] for x in win]
                    last_pred = mod.predict_on_batch(np.array(batch_x))
                mod.reset_states()

                r = model.pred_to_tags(last_pred, model.OUTPUTS[name])
                if r:
                    tags.append(r)

            if not tags:
                fd.write(' \n')
            else:
                fd.write(" ".join(tags) + '\n')

    pass
