"""microModel — SSL pretrain / train / infer package."""

import os
import warnings

# The micro conda env loads Intel OpenMP (torch/MKL) and LLVM OpenMP (numba,
# pulled in by umap/pacmap) into the same process. Intel OpenMP would abort on
# the duplicate and threadpoolctl warns on every run; opting into the standard
# KMP_DUPLICATE_LIB_OK workaround (runs complete fine) and silencing the
# repeated RuntimeWarning keeps the logs clean.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
warnings.filterwarnings("ignore", message="Found Intel OpenMP",
                        category=RuntimeWarning)

__version__ = "0.9.1"
