"""Run after installing the package: python examples/quadratic.py."""
from ppe.demo import make_demo

if __name__ == "__main__":
    trainer, loaders, interaction = make_demo("runs/quadratic-python")
    result = trainer.fit(*loaders, interaction=interaction)
    print(result.status, result.output_dir)
