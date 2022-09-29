#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import sys
import torch

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    elif "model" in obj:
        obj = obj["model"]
    else:
        raise Exception("Cannot find the model in the checkpoint.")
    
    new_model = {}
    for k, v in obj.items():
        if not k.startswith("module.encoder_k.") or "fc" in k:
            continue
        old_k = k
        k = k.replace("module.encoder_k.", "")
        print(old_k, "->", k)
        new_model[k] = v
    
    res = {"state_dict": new_model}

    torch.save(res, sys.argv[2])
