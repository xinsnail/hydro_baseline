"""Train and evaluate the HyDRO baseline used by the LAPT pipeline."""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from graphslim.config import cli
from graphslim.condensation.hydro import HyDRO
from graphslim.dataset import attack, get_dataset
from graphslim.dataset.utils import load_reduced
from graphslim.evaluation.paper_link_prediction import PaperLinkPredictionEvaluator
from graphslim.utils import seed_everything

if __name__ == "__main__":
    args = cli(standalone_mode=False)
    if not hasattr(args, "dataset"):
        sys.exit(args if isinstance(args, int) else 0)
    if args.method != "hydro":
        raise ValueError("This minimal bundle supports only -M hydro")
    if args.condensed_selection_metric != "lp":
        raise ValueError(
            "This LAPT bundle requires --condensed_selection_metric lp "
            "(the bundled dataset configuration supplies this value)."
        )
    seed_everything(args.seed)
    graph = get_dataset(args.dataset, args, args.load_path)
    if args.attack is not None and args.setting == "ind":
        graph = attack(graph, args)
    agent = HyDRO(
        setting=args.setting,
        reduction_rate=args.reduction_rate,
        data=graph,
        args=args,
    )
    agent.reduce(graph, verbose=args.verbose)
    graph.adj_syn, graph.feat_syn, graph.labels_syn = load_reduced(args, graph)
    PaperLinkPredictionEvaluator(args).evaluate(graph, reduced=True, verbose=True)
