import os
import sys
import pprint
import json

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util


ENTITY_MAPPING_FILE = "/home/ma/ma_ma/ma_leonenge/datasets/knowledge_graphs/entity.json"
RELATION_MAPPING_FILE = "/home/ma/ma_ma/ma_leonenge/datasets/knowledge_graphs/relation.json"


def load_vocab(dataset):
    with open(ENTITY_MAPPING_FILE, "r") as fin:
        entity_mapping = json.load(fin)
    with open(RELATION_MAPPING_FILE, "r") as fin:
        relation_mapping = json.load(fin)

    if hasattr(dataset, "test_entity_vocab"):
        entity_tokens = dataset.test_entity_vocab
    else:
        entity_tokens = dataset.entity_vocab
    relation_tokens = dataset.relation_vocab

    entity_vocab = [entity_mapping[token]["name"] for token in entity_tokens]
    relation_vocab = [relation_mapping[token]["name"] for token in relation_tokens]

    return entity_vocab, relation_vocab


def visualize_path(solver, triplet, entity_vocab, relation_vocab):
    num_relation = len(relation_vocab)
    h, t, r = triplet.tolist()
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    with torch.no_grad():
        pred, target = solver.model.predict_and_target(triplet)
    if isinstance(target, tuple):
        mask, target = target
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
        rankings = rankings.squeeze(0)
    else:
        pos_pred = pred.gather(-1, target.unsqueeze(-1))
        rankings = torch.sum(pos_pred <= pred, dim=-1) + 1

    logger.warning("")
    samples = (triplet, inverse)
    for sample, ranking in zip(samples, rankings):
        h, t, r = sample.squeeze(0).tolist()
        h_name = entity_vocab[h]
        t_name = entity_vocab[t]
        r_name = relation_vocab[r % num_relation]
        if r >= num_relation:
            r_name += "^(-1)"
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))

        paths, weights = solver.model.visualize(sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r in path:
                h_name = entity_vocab[h]
                t_name = entity_vocab[t]
                r_name = relation_vocab[r % num_relation]
                if r >= num_relation:
                    r_name += "^(-1)"
                triplets.append("<%s, %s, %s>" % (h_name, r_name, t_name))
            logger.warning("weight: %g\n\t%s" % (weight, " ->\n\t".join(triplets)))


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] != "WN18RRInductive":
        raise ValueError("Visualization is only implemented for %s" % cfg.dataset["class"])

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_vocab, relation_vocab = load_vocab(dataset)

    for i in range(min(500, len(solver.test_set))):
        visualize_path(solver, solver.test_set[i], entity_vocab, relation_vocab)
