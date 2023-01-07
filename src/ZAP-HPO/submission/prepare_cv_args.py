import os
import numpy as np

def prepare_cv_training_args(mode = "bpr", weighted = False, weigh_fn = "v0", use_meta = True, sparsity = 0.0, num_cv_folds = 5):

    base_cmd = " ".join(["--split_type cv",
                         "--mode", mode,
                         "--weighted", str(weighted),
                         "--weigh_fn", weigh_fn,
                         "--use_meta", str(use_meta),
                         "--sparsity", str(sparsity)])

    args = []
    for cv in range(1, num_cv_folds+1):
        cmd = " ".join([base_cmd, "--cv", str(cv)])
        args.append(cmd)

    return args

all_args = []

all_args += prepare_cv_training_args()
all_args += prepare_cv_training_args(use_meta = False)
all_args += prepare_cv_training_args(sparsity = 0.75)
all_args += prepare_cv_training_args(sparsity = 0.50)
all_args += prepare_cv_training_args(sparsity = 0.25)
all_args += prepare_cv_training_args(weighted = True, weigh_fn = "v0")
all_args += prepare_cv_training_args(weighted = True, weigh_fn = "v1")
all_args += prepare_cv_training_args(mode = "regression")
all_args += prepare_cv_training_args(mode = "regression", use_meta = False)
all_args += prepare_cv_training_args(mode = "regression", weighted = True)
all_args += prepare_cv_training_args(mode = "tml")
all_args += prepare_cv_training_args(mode = "tml", use_meta = False)

savepath = "./submission/cv.args"
f = open(savepath, "w+")
f.write("\n".join(all_args))
f.close()

