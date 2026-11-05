import os, sys, collections, atexit
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm.v1.worker.gpu import cudagraph_utils as cu
C = collections.Counter()
orig = cu.CudaGraphManager.dispatch
def dispatch(self, num_reqs, num_tokens, uniform_token_count, *a, **k):
    d = orig(self, num_reqs, num_tokens, uniform_token_count, *a, **k)
    C[(type(self).__name__, num_tokens, uniform_token_count, str(d.cg_mode).split(".")[-1].split(":")[0])] += 1
    return d
cu.CudaGraphManager.dispatch = dispatch
atexit.register(lambda: [print("CG", k, v) for k, v in C.most_common(20)])
sys.argv = [sys.argv[0]] + sys.argv[1:]
import runpy
runpy.run_path(os.path.join(os.path.dirname(__file__), "gen_spec.py"), run_name="__main__")
