import csv
import os
import torch
import logging
import sys

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name):
    return os.path.join(get_storage_dir(), model_name)


def get_status_path(model_dir, best = False):
    if best:
        return os.path.join(model_dir, "best.pt")
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, map_location = None, best = False):
    path = get_status_path(model_dir, best)
    if map_location is not None:
        return torch.load(path, map_location = map_location)
    return torch.load(path)



def save_status(status, model_dir, best = False):
    path = get_status_path(model_dir, best)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir, map_location = None, best = False):
    if map_location is not None:
        return get_status(model_dir, map_location, best)["vocab"]
    return get_status(model_dir, best)["vocab"]


def get_model_state(model_dir, map_location = None, best = False):
    if map_location is not None:
        return get_status(model_dir, map_location, best)["model_state"]
    return get_status(model_dir, best)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)
