import os
import sys
import json
import pprint
import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util

entity_mapping_file = "/home/ma/ma_ma/ma_leonenge/datasets/knowledge_graphs/entities.json"
relation_mapping_file = "/home/ma/ma_ma/ma_leonenge/datasets/knowledge_graphs/relations.json"

def load_entity_mapping(entity_mapping_file):
    with open(entity_mapping_file, "r") as fin:
        entity_mapping = json.load(fin)
    return entity_mapping

def load_relation_mapping(relation_mapping_file):
    with open(relation_mapping_file, "r") as fin:
        relation_mapping = json.load(fin)
    return relation_mapping

def visualize_path(solver, triplet, entity_mapping, relation_mapping):
    num_relation = len(relation_mapping)
    h, t, r = triplet.tolist()
    triplet = torch.as_tensor([[h, t, r]], device=solver.device)
    inverse = torch.as_tensor([[t, h, r + num_relation]], device=solver.device)
    solver.model.eval()
    pred, (mask, target) = solver.model.predict_and_target(triplet)
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    rankings = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    rankings = rankings.squeeze(0)

    logger.warning("")
    samples = (triplet, inverse)
    for sample, ranking in zip(samples, rankings):
        h, t, r = sample.squeeze(0).tolist()
        
        # Zugriff auf Entität und Relation über das Mapping
        h_name = entity_mapping[str(h)]["name"]  # Entität-Name durch ID bekommen
        h_desc = entity_mapping[str(h)]["desc"]  # Entität-Beschreibung durch ID bekommen
        t_name = entity_mapping[str(t)]["name"]
        t_desc = entity_mapping[str(t)]["desc"]
        
        # Relationen abrufen, sowohl Name als auch Beschreibung
        r_name = relation_mapping[list(relation_mapping.keys())[r]]["name"]
        r_desc = relation_mapping[list(relation_mapping.keys())[r]].get("desc", "No description available")
        
        logger.warning(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logger.warning("rank(%s | %s, %s) = %g" % (t_name, h_name, r_name, ranking))
        logger.warning("Entity %s: %s" % (h_name, h_desc))
        logger.warning("Entity %s: %s" % (t_name, t_desc))
        logger.warning("Relation %s: %s" % (r_name, r_desc))

        paths, weights = solver.model.visualize(sample)
        for path, weight in zip(paths, weights):
            triplets = []
            for h, t, r in path:
                h_name = entity_mapping[str(h)]["name"]
                t_name = entity_mapping[str(t)]["name"]
                r_name = relation_mapping[list(relation_mapping.keys())[r]]["name"]
                r_desc = relation_mapping[list(relation_mapping.keys())[r]].get("desc", "No description available")
                triplets.append("<%s, %s (%s), %s>" % (h_name, r_name, r_desc, t_name))
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
        raise ValueError("Visualization is only implemented for WN18RR")

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    entity_mapping = load_entity_mapping(entity_mapping_file)
    relation_mapping = load_relation_mapping(relation_mapping_file)

    for i in range(500):
        visualize_path(solver, solver.test_set[i], entity_mapping, relation_mapping)






