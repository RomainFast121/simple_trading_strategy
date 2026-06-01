from pathlib import Path
import json
import re
import shutil

import pandas as pd

from utils import (
    add_close_return_targets,
    build_close_panel,
    build_xgb_features,
    extract_close_map,
    fetch_data as fetch_market_data,
    infer_xgb_feature_columns,
    make_xgb_model_frame,
)


def save_frame(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path)
        return path
    except Exception:
        if path.exists():
            path.unlink()
        fallback = path.with_suffix(".pkl")
        frame.to_pickle(fallback)
        return fallback


def load_frame(path):
    path = Path(path)
    if path.exists():
        if path.suffix == ".pkl":
            return pd.read_pickle(path)
        return pd.read_parquet(path)
    fallback = path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(path)


def safe_run_name(symbols, start, end, tf, hour=None, hour_timezone="UTC"):
    return "current"


def config_run_name(symbols, start, end, tf, hour=None, hour_timezone="UTC"):
    raw = "_".join(
        [
            "-".join(str(symbol) for symbol in symbols),
            str(start),
            str(end),
            str(tf),
            "hour-none" if hour is None else f"hour-{hour}",
            str(hour_timezone),
        ]
    )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_").lower()


def output_dir_for(run_name, root="local_outputs/xgb"):
    path = Path(root) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_generated_outputs(output_dir):
    output_dir = Path(output_dir)
    for name in [
        "feature_panel.parquet",
        "feature_panel.pkl",
        "model_frame.parquet",
        "model_frame.pkl",
        "feature_columns.json",
        "predictions.parquet",
        "predictions.pkl",
        "predictions.csv",
        "walkforward_metrics.csv",
        "parameter_schedule.csv",
        "parameter_sweep.csv",
        "strategy_data.csv",
        "buy_and_hold_data.csv",
        "summary.csv",
        "feature_ablation.csv",
        "xgb_run_diagnostics.json",
    ]:
        path = output_dir / name
        if path.exists():
            path.unlink()
    figures = output_dir / "figures"
    if figures.exists():
        shutil.rmtree(figures)


def data_signature(strategy):
    return {
        "ticker": list(strategy.tickers),
        "crypto": list(strategy.crypto_tickers),
        "symbols": list(strategy.symbols),
        "start": str(strategy.start),
        "end": str(strategy.end),
        "tf": str(strategy.tf),
        "hour": strategy.hour,
        "hour_timezone": str(strategy.hour_timezone),
    }


def feature_signature(strategy):
    return {
        **data_signature(strategy),
        "feature_builder": "close_xgb_v1",
    }


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_json(payload, path):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def _diagnostics_path(output_dir):
    return Path(output_dir) / "xgb_run_diagnostics.json"


def read_cache_signature(output_dir, key):
    payload = _read_json(_diagnostics_path(output_dir))
    return payload.get("cache", {}).get(key)


def write_cache_signature(output_dir, key, signature):
    path = _diagnostics_path(output_dir)
    payload = _read_json(path)
    cache = payload.setdefault("cache", {})
    cache[key] = signature
    _write_json(payload, path)


def fetch_or_load_raw_data(strategy, force=False, save=True):
    if not force:
        cached = load_raw_data(strategy.output_dir, strategy.symbols, expected_signature=data_signature(strategy))
        if cached:
            strategy.raw_data = cached
            print(f"Loaded cached raw data from {strategy.output_dir}.")
            return cached

    data = fetch_market_data(
        ticker=strategy.tickers,
        crypto=strategy.crypto_tickers,
        start=strategy.start,
        end=strategy.end,
        interval=strategy.tf,
        auto_adjust=True,
        progress=False,
        hour=strategy.hour,
        hour_timezone=strategy.hour_timezone,
    )
    strategy.raw_data = data
    if save:
        clear_generated_outputs(strategy.output_dir)
        save_raw_data(data, strategy.output_dir)
        write_cache_signature(strategy.output_dir, "data", data_signature(strategy))
    return data


def _symbol_path(output_dir, symbol):
    safe_symbol = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(symbol)).strip("_").lower()
    return Path(output_dir) / "raw" / f"{safe_symbol}.parquet"


def save_raw_data(raw_data, output_dir):
    raw_dir = Path(output_dir) / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for symbol, frame in raw_data.items():
        path = _symbol_path(output_dir, symbol)
        path = save_frame(frame, path)
        manifest.append({"symbol": symbol, "path": str(path.relative_to(output_dir))})
    pd.DataFrame(manifest).to_csv(Path(output_dir) / "raw_manifest.csv", index=False)


def load_raw_data(output_dir, symbols, expected_signature=None):
    if expected_signature is not None and read_cache_signature(output_dir, "data") != expected_signature:
        return {}
    manifest_path = Path(output_dir) / "raw_manifest.csv"
    if not manifest_path.exists():
        return {}
    manifest = pd.read_csv(manifest_path)
    data = {}
    for _, row in manifest.iterrows():
        symbol = row["symbol"]
        path = Path(output_dir) / row["path"]
        if symbol in symbols and path.exists():
            data[symbol] = load_frame(path)
    return data if len(data) == len(symbols) else {}


def build_or_load_feature_panel(strategy, force=False, save=True):
    feature_path = Path(strategy.output_dir) / "feature_panel.parquet"
    model_path = Path(strategy.output_dir) / "model_frame.parquet"
    columns_path = Path(strategy.output_dir) / "feature_columns.json"
    expected_signature = feature_signature(strategy)

    feature_exists = feature_path.exists() or feature_path.with_suffix(".pkl").exists()
    model_exists = model_path.exists() or model_path.with_suffix(".pkl").exists()
    signature_matches = read_cache_signature(strategy.output_dir, "features") == expected_signature
    if not force and signature_matches and feature_exists and model_exists and columns_path.exists():
        strategy.feature_panel = load_frame(feature_path)
        strategy.model_frame = load_frame(model_path)
        strategy.feature_columns = pd.read_json(columns_path, typ="series").tolist()
        return strategy.feature_panel

    if not strategy.raw_data:
        cached = load_raw_data(strategy.output_dir, strategy.symbols, expected_signature=data_signature(strategy))
        if cached:
            strategy.raw_data = cached
        else:
            raise ValueError("No raw data in memory or cache. Run fetch_data() first.")

    close_map = extract_close_map(strategy.raw_data, symbols=strategy.symbols)
    panel = build_close_panel(close_map, symbols=strategy.symbols)
    panel = add_close_return_targets(panel, drop_incomplete=False)
    panel = build_xgb_features(panel)

    feature_columns = infer_xgb_feature_columns(panel)
    model_frame = make_xgb_model_frame(panel, feature_columns)

    strategy.feature_panel = panel
    strategy.model_frame = model_frame
    strategy.feature_columns = feature_columns

    if save:
        save_frame(panel, feature_path)
        save_frame(model_frame, model_path)
        pd.Series(feature_columns).to_json(columns_path)
        write_cache_signature(strategy.output_dir, "features", expected_signature)
    return panel
