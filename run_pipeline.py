"""Quick Start Runner - python run_pipeline.py --mode test --build-graph"""
import argparse, logging, sys, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test","sp500","kospi","custom"], default="test")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--steps", nargs="+", default=None)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--target-return", type=float, default=0.10)
    parser.add_argument("--stop-loss", type=float, default=-0.05)
    parser.add_argument("--build-graph", action="store_true")
    args = parser.parse_args()
    from config import get_test_config, get_sp500_config, get_kospi_config, PipelineConfig
    cfg = {"test":get_test_config,"sp500":get_sp500_config,"kospi":get_kospi_config}.get(args.mode, get_test_config)()
    if args.mode == "custom":
        cfg = PipelineConfig(); cfg.data.universe="custom"; cfg.data.custom_tickers=args.tickers or ["AAPL","MSFT"]
    cfg.data.start_date = args.start_date
    cfg.survival.target_return = args.target_return
    cfg.survival.stop_loss = args.stop_loss
    logger.info(f"Mode={args.mode} Target={cfg.survival.target_return:+.0%} Stop={cfg.survival.stop_loss:+.0%}")
    from data_pipeline import DataPipeline
    results = DataPipeline(cfg).run(steps=args.steps)
    import pandas as pd
    for k,v in results.items():
        if isinstance(v,pd.DataFrame): print(f"  {k:25s}: {v.shape}")
        elif isinstance(v,(dict,list)): print(f"  {k:25s}: {len(v)} items")
    if args.build_graph and "prices" in results and "sector_map" in results:
        from graph_builder import build_full_graph
        _,gd = build_full_graph(results["sector_map"],results.get("prices_with_ta",results["prices"]),results["tickers"],cfg)
        print(f"Graph: {gd['num_nodes']} nodes, {gd['num_edges']} edges")
    if "survival_labels" in results:
        l=results["survival_labels"]
        print(f"Labels: {len(l):,} total | Profit:{(l['event_type']==1).mean()*100:.1f}% Loss:{(l['event_type']==2).mean()*100:.1f}%")

if __name__=="__main__": main()
