# -*- coding: utf-8 -*-
# Copyright (C) Huawei Technologies Co., Ltd. 2019.
"""

"""
import json


def read_objs4json_file(file_name):
    """
    Read a list of object from a json file, each line is loaded as one object.
    :param file_name: json file name
    :return: list of object
    """
    with open(file_name, "r", encoding="utf-8") as reader:
        for line in reader:
            yield json.loads(line)


def write_objs2json_file(objs, file_name):
    """
    Write a list of object to a json file, each object is dumped as one line.
    :param objs: list of object
    :param file_name: json file name
    :return: None
    """
    with open(file_name, "w",
              encoding="utf-8") as writer:
        for obj in objs:
            writer.write(json.dumps(obj, ensure_ascii=False))
            writer.write("\n")
