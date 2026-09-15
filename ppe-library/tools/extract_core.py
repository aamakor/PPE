"""Generate PPE numerical definitions from src/function.py.

Run from any directory: python ppe-library/tools/extract_core.py
The training orchestration in src/ppe/_training.py is maintained directly,
including archive snapshots and recovery. This tool only writes _numerics.py
and source_manifest.json. Review numerical changes before accepting them.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "ppe-library/src/ppe"


class Adapter(ast.NodeTransformer):
    def __init__(self, training=False):
        self.training = training
        self.in_class = False

    def visit_ClassDef(self, node):
        self.in_class = True
        node = self.generic_visit(node)
        self.in_class = False
        return node

    def visit_Name(self, node):
        if not self.training and node.id == "device" and isinstance(node.ctx, ast.Load):
            return ast.parse("next(self.model.parameters()).device" if self.in_class else
                             "next(model.parameters()).device", mode="eval").body
        return node

    def visit_Call(self, node):
        source = ast.unparse(node)
        if source in ("criterion(outs[i], labels[:, i].long())",
                      "self.criterion(outs[i], labels[:, i].long())"):
            owner = "self.criterion" if self.in_class else "criterion"
            return ast.parse(f"{owner}.task_loss(i, outs[i], labels)", mode="eval").body
        if source == "labels.size(1)":
            return ast.parse("self.criterion.n_objectives" if self.in_class else
                             "criterion.n_objectives", mode="eval").body
        if source == "labels.to(device)":
            node = ast.parse("move_targets(labels, device)", mode="eval").body
        if source == "topk_accuracies(outs[i], labels[:, i].long())[0]":
            raise AssertionError("subscript should be handled first")
        if self.training:
            replacements = {
                "torch.cuda.synchronize()": "runtime.synchronize()",
                "get_preference(nobj)": "runtime.preference(nobj)",
                "dm_desire()": "runtime.interaction.accept_front()",
                "request_user_action(pref, num_pred)": "runtime.action(pref, num_pred)",
                "float(input(f'Enter new step size '))": "runtime.step_size()",
            }
            if source in replacements:
                return ast.parse(replacements[source], mode="eval").body
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        if ast.unparse(node) == "topk_accuracies(outs[i], labels[:, i].long())[0]":
            return ast.parse("criterion.metric(i, outs[i], labels)", mode="eval").body
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if self.training and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) == "ax.scatter":
                return ast.parse("runtime.record_initial(init_losses, test_ilosses, criterion.names)").body[0]
            if ast.unparse(node.value.func).startswith(("plt.", "ax.")):
                return ast.Pass()
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if self.training:
            name = ast.unparse(node.targets[0])
            if name in ("fig", "ax"):
                return ast.Pass()
            if name == "file_path":
                node.value = ast.parse("runtime.results_dir", mode="eval").body
        return self.generic_visit(node)

    def visit_Try(self, node):
        if self.training and "FILE_DIR" in ast.unparse(node):
            return ast.Pass()
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if self.training and ast.unparse(node.test) in ("type == 'UCI'", "type == 'INT'"):
            return ast.Pass()
        return node


def generate():
    source = ast.parse((ROOT / "src/function.py").read_text())
    # Pure definitions are copied. Dataset imports, global seeds/device selection,
    # terminal helpers and unused cvxopt imports are not library dependencies.
    omit = {"get_preference", "dm_desire", "request_user_action", "get_file_directory"}
    definitions = []
    for node in source.body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)) or node.name in omit:
            continue
        if node.name == "proj_general":  # this helper has its own local device
            definitions.append(node)
        else:
            definitions.append(Adapter().visit(node))
    # Archive metadata arity only: comparisons are unchanged; return original tuple.
    for node in definitions:
        if node.name == "prune_archive":
            text = ast.unparse(node)
            text = text.replace("(v_i, p_i, d_i, acc_i)", "entry_i")
            text = text.replace("(v_j, p_j, d_j, acc_j)", "entry_j")
            parsed = ast.parse(text).body[0]
            for loop in ast.walk(parsed):
                if isinstance(loop, ast.For) and isinstance(loop.target, ast.Tuple):
                    name = loop.target.elts[-1]
                    if isinstance(name, ast.Name) and name.id in {"entry_i", "entry_j"}:
                        suffix = name.id[-1]
                        loop.body.insert(0, ast.parse(f"v_{suffix} = entry_{suffix}[0]").body[0])
            node.body = parsed.body
    imports = '''from itertools import combinations
from pathlib import Path
from contextlib import contextmanager
import shutil
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, minres
from scipy.optimize import nnls
from tqdm import tqdm, trange
from .objectives import move_targets
from .runtime import get_random_batch
'''
    header = '# Generated by tools/extract_core.py; algorithm changes require creator review.\n'
    DEST.joinpath("_numerics.py").write_text(header + imports + "\n\n" +
        ast.unparse(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))) + "\n")

    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
              for p in ("src/function.py", "src/util.py", "Data/mnist_data.py")}
    ROOT.joinpath("ppe-library/source_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    generate()
