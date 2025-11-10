#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


@st.cache_data
def load_symbol_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    p = Path("results/liquidity_present_pairs.csv")
    if p.exists():
        try:
            df = pd.read_csv(p)
            if {"from_address", "symbol"}.issubset(df.columns):
                for _, r in df.iterrows():
                    addr = str(r["from_address"]).lower()
                    sym = str(r["symbol"]).upper()
                    mapping[addr] = sym
        except Exception:
            pass
    mapping.setdefault("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "WBTC")
    mapping.setdefault("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "WETH")
    return mapping


EXCLUDE_SYMBOLS = {"WM", "TBILL"}


def _filter_assets(addresses: list[str]) -> list[str]:
    sym_map = load_symbol_map()
    out: list[str] = []
    for a in addresses:
        sym = sym_map.get(str(a).lower())
        if sym and sym.upper() in EXCLUDE_SYMBOLS:
            continue
        out.append(a)
    return out


def _pretty_level(name: str) -> str:
    """Render directory-style level names as human-friendly text.
    Examples: '+0p10' -> '+0.10', 'm0p15' -> '-0.15', '0p05' -> '0.05'.
    """
    s = str(name)
    # Replace minus marker and decimal marker
    s = s.replace("m", "-").replace("p", ".")
    # Ensure '+0.10' stays as is; others already fine
    return s


@st.cache_data
def discover_results_roots(base: str = "results") -> list[str]:
    """Find results roots that contain a dashboard/runs.csv."""
    roots: list[str] = []
    base_path = Path(base)
    if not base_path.exists():
        return roots
    for p in base_path.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("beta_suite"):
            continue
        if (p / "dashboard" / "runs.csv").exists():
            roots.append(str(p))
    # Ensure stable order: full, fine grids, smoke last if present
    roots = sorted(roots)
    return roots


@st.cache_data
def load_runs(root: Path) -> pd.DataFrame:
    p = root / "dashboard" / "runs.csv"
    if not p.exists():
        st.error(f"Missing runs table: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Normalize asset id casing
    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.lower()
    # Coerce numeric
    for c in [
        "shock",
        "tvl_initial",
        "tvl_final",
        "total_liquidations",
        "total_liquidated_value_usd",
        "final_bad_debt_usd",
        "liquidation_efficiency",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data
def load_ub_events(root: Path) -> pd.DataFrame:
    p = root / "dashboard" / "ub_events.csv"
    if not p.exists():
        return pd.DataFrame(
            columns=["asset", "shock", "user_id", "liquidated_value_usd_env", "bad_debt_usd_env"]
        )
    df = pd.read_csv(p)
    if "asset" in df.columns:
        df["asset"] = df["asset"].astype(str).str.lower()
    for c in ["shock", "liquidated_value_usd_env", "bad_debt_usd_env"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data
def load_ev_probs(root: Path) -> dict:
    p = Path("results/config/shock_probabilities.json")
    if not p.exists():
        return {
            "grid": [-0.05, -0.10, -0.20, -0.30, -0.40],
            "assets": {},
            "default": [0.30, 0.25, 0.20, 0.15, 0.10],
            "used_default": {},
            "note": "default",
        }
    try:
        return json.loads(p.read_text())
    except Exception:
        return {
            "grid": [-0.05, -0.10, -0.20, -0.30, -0.40],
            "assets": {},
            "default": [0.30, 0.25, 0.20, 0.15, 0.10],
            "used_default": {},
            "note": "default",
        }


def compute_ev_bad_debt(df_runs: pd.DataFrame, ev: dict) -> pd.DataFrame:
    # Agent mode only; pivot shocks
    if df_runs.empty:
        return pd.DataFrame(columns=["asset", "ev_bad_debt"])
    agent = df_runs[df_runs["mode"] == "Agent"].copy()
    if agent.empty:
        return pd.DataFrame(columns=["asset", "ev_bad_debt"])
    grid = ev.get("grid", [])
    used_default = ev.get("used_default", {})
    default = ev.get("default", [])
    out = []
    for asset, gdf in agent.groupby("asset"):
        # map shocks to bad debt
        bd_map = {
            round(float(s), 4): float(b) if pd.notna(b) else 0.0
            for s, b in zip(gdf["shock"], gdf["final_bad_debt_usd"], strict=False)
        }
        probs = ev.get("assets", {}).get(asset, default)
        if len(probs) != len(grid):
            probs = default
        ev_bd = 0.0
        for p, m in zip(probs, grid, strict=False):
            ev_bd += float(p) * float(bd_map.get(round(float(m), 4), 0.0))
        out.append(
            {
                "asset": asset,
                "ev_bad_debt": ev_bd,
                "used_default": bool(used_default.get(asset, False)),
            }
        )
    return pd.DataFrame(out).sort_values("ev_bad_debt", ascending=False)


def _filter_shocks_df(df: pd.DataFrame, step: float) -> pd.DataFrame:
    if df.empty or "shock" not in df.columns:
        return df
    # Keep shocks that are multiples of step (tolerance for floats)
    tol = 1e-9
    mask = abs((df["shock"].abs() / step) - (df["shock"].abs() / step).round()) <= tol
    return df[mask].copy()


@st.cache_data
def load_reserve_symbol_map(reserves_csv: str = "data/euler/reserves.csv") -> dict[str, str]:
    mapping: dict[str, str] = {}
    p = Path(reserves_csv)
    if not p.exists():
        return mapping
    try:
        df = pd.read_csv(p, dtype=str)
        if {"reserve_id", "asset_symbol"}.issubset(df.columns):
            for _, r in df.iterrows():
                mapping[str(r["reserve_id"]) ] = str(r["asset_symbol"]).upper()
    except Exception:
        pass
    return mapping


@st.cache_data
def load_ub_event_json(root: Path, asset: str, shock: float) -> list[dict]:
    key = f"{asset}_m{int(abs(shock)*100):03d}"
    p = root / "ub" / key / "events.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


@st.cache_data
def load_agent_event_json(root: Path, asset: str, shock: float) -> list[dict]:
    key = f"{asset}_m{int(abs(shock)*100):03d}"
    p = root / "agent" / key / "events.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _pick_agent_events_root(current_root: Path, asset: str, shock: float) -> tuple[Path, bool]:
    """Return a results root that has Agent events for the selected asset/shock.

    Only affects Breakdown (Agent). Other pages continue using current_root.
    """
    preferred = [
        current_root,
        Path("results/beta_suite_fine_0025"),
        Path("results/beta_suite_fine_005"),
        Path("results/beta_suite_full"),
    ]
    seen: set[str] = set()
    key = f"{asset}_m{int(abs(float(shock))*100):03d}"
    for r in preferred:
        rp = Path(r)
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        if (rp / "agent" / key / "events.json").exists():
            return rp, (rp != current_root)
    return current_root, False


def _aggregate_by_reserve(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    coll_rows: list[dict] = []
    debt_rows: list[dict] = []
    for e in events:
        usd = float(e.get("liquidated_value_usd_env", 0.0) or 0.0)
        uid = str(e.get("user_id", "")).lower()
        for rid, _amt in (e.get("collateral_reserves") or {}).items():
            coll_rows.append({"reserve_id": str(rid), "user_id": uid, "usd": usd})
        for rid, _amt in (e.get("debt_reserves") or {}).items():
            debt_rows.append({"reserve_id": str(rid), "user_id": uid, "usd": usd})

    def _agg(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["reserve_id", "users", "total_liquidated_usd_env"])
        d = pd.DataFrame(rows)
        users = d.groupby("reserve_id")["user_id"].nunique().rename("users")
        totals = d.groupby("reserve_id")["usd"].sum().rename("total_liquidated_usd_env")
        out = pd.concat([users, totals], axis=1).reset_index()
        # Attach symbols if available
        rmap = load_reserve_symbol_map()
        out["reserve_symbol"] = out["reserve_id"].map(lambda r: rmap.get(str(r), str(r)))
        return out.sort_values("total_liquidated_usd_env", ascending=False)

    return _agg(coll_rows), _agg(debt_rows)


@st.cache_data
def load_prices_wide(prices_csv: str = "data/prices/historical_prices_365d.csv") -> pd.DataFrame:
    p = Path(prices_csv)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "chain_id" in df.columns and {"address", "price"}.issubset(df.columns):
        idx = "timestamp" if "timestamp" in df.columns else None
        if idx:
            if df.duplicated(subset=[idx, "address"]).any():
                df = df.sort_values([idx]).groupby([idx, "address"], as_index=False).last()
            wide = df.pivot(index=idx, columns="address", values="price").sort_index()
        else:
            wide = df.pivot(columns="address", values="price")
    else:
        wide = df
    wide.columns = [str(c).lower() for c in wide.columns]
    return wide


@st.cache_data
def load_corr_matrix(corr_csv: str = "data/prices/corr_365d.csv") -> pd.DataFrame:
    p = Path(corr_csv)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.map(lambda x: str(x).lower())
    df.columns = [str(c).lower() for c in df.columns]
    return df


# Curated list of relevant assets for snippet/example displays (address -> label order)
def _relevant_assets() -> list[tuple[str, str]]:
    return [
        ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "WETH"),
        ("0xbf5495efe5db9ce00f80364c8b423567e58d2110", "ezETH"),
        ("0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee", "weETH"),
        ("0xae78736cd615f374d3085123a210448e74fc6393", "rETH"),
        ("0xbe9895146f7af43049ca1c1ae358b0541ea49704", "cbETH"),
        ("0xa1290d69c65a6fe4df752f95823fae25cb99e5a7", "rsETH"),
        ("0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0", "wstETH"),
        ("0xd11c452fc99cf405034ee446803b6f6c1f6d5ed8", "tETH"),
        ("0x8236a87084f8b84306f72007f36f2618a5634494", "LBTC"),
        ("0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf", "cbBTC"),
        ("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "WBTC"),
        ("0xa3931d71877c0e7a3148cb7eb4463524fec27fbd", "sUSDS"),
        ("0x8292bb45bf1ee4d140127049757c2e0ff06317ed", "RLUSD"),
        ("0x4c9edd5852cd905f086c759e8383e09bff1e68b3", "USDe"),
        ("0x68749665ff8d2d112fa859aa293f07a622782f38", "XAUt"),
        ("0x9d39a5de30e57443bff2a8307a4256c8797a3497", "sUSDe"),
        ("0xc139190f447e929f090edeb554d95abb8b18ac1c", "USDtb"),
        ("0xdc035d45d973e3ec169d2276ddab16f1e407384f", "USDS"),
        ("0x80ac24aa929eaf5013f6436cda2a7ba190f5cc0b", "syrupUSDC"),
        ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "USDC"),
        ("0xdac17f958d2ee523a2206206994597c13d831ec7", "USDT"),
        ("0x437cc33344a0b27a429f795ff6b469c72698b291", "wM"),
    ]


def page_overview(root: Path, shock_step: float):
    st.header("Overview")
    df_runs = load_runs(root)
    if df_runs.empty:
        st.info("No runs found.")
        return
    sym_map = load_symbol_map()
    assets = sorted(df_runs["asset"].astype(str).str.lower().unique().tolist())
    assets = _filter_assets(assets)
    asset = st.selectbox("Asset", assets, format_func=lambda a: sym_map.get(str(a).lower(), str(a)))
    sdf = df_runs[df_runs["asset"] == asset].copy()
    sdf = _filter_shocks_df(sdf, shock_step)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            sdf,
            x="shock",
            y="total_liquidated_value_usd",
            color="mode",
            title="Total Liquidated USD vs Shock",
        )
        st.plotly_chart(fig, width="stretch")
        if not sdf.empty:
            fig.update_xaxes(tickvals=sorted(sdf["shock"].unique()))
    with col2:
        _agent = sdf[sdf["mode"] == "Agent"].copy()
        fig = px.line(_agent, x="shock", y="final_bad_debt_usd", title="Bad Debt (Agent) vs Shock")
        st.plotly_chart(fig, width="stretch")
        if not _agent.empty:
            fig.update_xaxes(tickvals=sorted(_agent["shock"].unique()))

    # EV
    st.subheader("Expected Bad Debt (EV)")
    ev = load_ev_probs(root)
    df_ev = compute_ev_bad_debt(df_runs, ev)
    # Map asset addresses to symbols for display
    sym_map = load_symbol_map()
    if not df_ev.empty and "asset" in df_ev.columns:
        df_ev["asset"] = df_ev["asset"].astype(str).str.lower().map(lambda a: sym_map.get(a, a))
    if df_ev.empty:
        st.info("No EV data.")
    else:
        if "used_default" in df_ev.columns:
            styled = df_ev.style.apply(
                lambda row: [
                    "background-color: #ffecec" if bool(row.get("used_default", False)) else ""
                    for _ in row.index
                ],
                axis=1,
            )
            st.dataframe(styled, width="stretch")
        else:
            st.dataframe(df_ev, width="stretch")
        if "note" in ev:
            st.caption(
                f"EV source: {ev.get('note','')}. Red rows indicate default probabilities were used."
            )


def page_asset_details(root: Path, shock_step: float):
    st.header("Asset Details")
    df_runs = load_runs(root)
    if df_runs.empty:
        st.info("No runs found.")
        return
    sym_map = load_symbol_map()
    assets = sorted(df_runs["asset"].astype(str).str.lower().unique().tolist())
    assets = _filter_assets(assets)
    asset = st.selectbox(
        "Asset",
        assets,
        key="asset_details_asset",
        format_func=lambda a: sym_map.get(str(a).lower(), str(a)),
    )
    sdf = df_runs[df_runs["asset"] == asset].copy()
    sdf = _filter_shocks_df(sdf, shock_step)
    # Table of scenarios
    st.subheader("Scenario Table (UB & Agent)")
    # Replace asset address with symbol for display
    sym_map = load_symbol_map()
    if "asset" in sdf.columns:
        sdf["asset"] = sdf["asset"].astype(str).str.lower().map(lambda a: sym_map.get(a, a))
    st.dataframe(sdf.sort_values(["shock", "mode"]), width="stretch")


def page_breakdown_agent(root: Path):
    st.header("Breakdown (Agent)")
    df_runs = load_runs(root)
    if df_runs.empty:
        st.info("No runs found.")
        return
    agent = df_runs[df_runs["mode"] == "Agent"].copy()
    if agent.empty:
        st.info("No Agent runs present in this results folder.")
        return
    sym_map = load_symbol_map()
    assets = sorted(agent["asset"].astype(str).str.lower().unique().tolist())
    assets = _filter_assets(assets)
    asset = st.selectbox(
        "Asset",
        assets,
        key="breakdown_asset",
        format_func=lambda a: sym_map.get(str(a).lower(), str(a)),
    )
    shocks = sorted(agent[agent["asset"] == asset]["shock"].unique().tolist())
    shock = st.selectbox("Shock", shocks, key="breakdown_shock")
    # Prefer a root that actually has Agent events for this asset/shock
    effective_root, switched = _pick_agent_events_root(root, asset, float(shock))
    events = load_agent_event_json(effective_root, asset, float(shock))
    if events:
        coll_df, debt_df = _aggregate_by_reserve(events)
        st.subheader("Collateral-side breakdown")
        st.caption("Users = unique wallets per reserve. USD totals can overcount if an event touches multiple reserves.")
        st.dataframe(coll_df, width="stretch")
        st.subheader("Debt-side breakdown")
        st.caption("Users = unique wallets per reserve. USD totals can overcount if an event touches multiple reserves.")
        st.dataframe(debt_df, width="stretch")
        # Per-user table
        users_df = pd.DataFrame([
            {
                "user_id": e.get("user_id"),
                "liquidated_value_usd": float(e.get("liquidated_value_usd_env", 0.0) or 0.0),
                "bad_debt_usd": float(e.get("bad_debt_usd_env", 0.0) or 0.0),
            }
            for e in events
        ])
        st.subheader("Per-user liquidations (Agent)")
        st.write(f"Rows: {len(users_df)}")
        st.dataframe(users_df.sort_values("liquidated_value_usd", ascending=False), width="stretch")
        if switched:
            st.caption(f"Loaded Agent events from {effective_root}")
    else:
        row = agent[(agent["asset"] == asset) & (agent["shock"] == shock)].copy()
        if row.empty:
            st.info("No Agent data for selected asset/shock.")
            return
        st.subheader("Scenario metrics (Agent)")
        view = row[[
            "asset",
            "shock",
            "total_liquidations",
            "total_liquidated_value_usd",
            "final_bad_debt_usd",
            "liquidation_efficiency",
        ]].copy()
        view["asset"] = view["asset"].astype(str).str.lower().map(lambda a: sym_map.get(a, a))
        st.dataframe(view.sort_values(["shock"]).reset_index(drop=True), width="stretch")
        st.caption("Agent per-user events not found for this scenario. Re-run the suite with the updated runner to persist Agent per-user events.")


def page_breakdown_ub(root: Path):
    st.header("Breakdown (UB)")
    df_runs = load_runs(root)
    df_ub = load_ub_events(root)
    if df_runs.empty or df_ub.empty:
        st.info("No UB per-user events found.")
        return
    sym_map = load_symbol_map()
    assets = sorted(df_ub["asset"].astype(str).str.lower().unique().tolist())
    assets = _filter_assets(assets)
    asset = st.selectbox(
        "Asset",
        assets,
        key="breakdown_ub_asset",
        format_func=lambda a: sym_map.get(str(a).lower(), str(a)),
    )
    shocks = sorted(df_ub[df_ub["asset"] == asset]["shock"].unique().tolist())
    shock = st.selectbox("Shock", shocks, key="breakdown_ub_shock")
    events = load_ub_event_json(root, asset, float(shock))
    coll_df, debt_df = _aggregate_by_reserve(events)
    st.subheader("Collateral-side breakdown")
    st.caption("Users = unique wallets per reserve. USD totals can overcount if an event touches multiple reserves.")
    st.dataframe(coll_df, width="stretch")
    st.subheader("Debt-side breakdown")
    st.caption("Users = unique wallets per reserve. USD totals can overcount if an event touches multiple reserves.")
    st.dataframe(debt_df, width="stretch")
    sdf = df_ub[(df_ub["asset"] == asset) & (df_ub["shock"] == shock)].copy()
    if "asset" in sdf.columns:
        sym_map = load_symbol_map()
        sdf["asset"] = sdf["asset"].astype(str).str.lower().map(lambda a: sym_map.get(a, a))
    st.subheader("Per-user liquidations (UB)")
    st.write(f"Rows: {len(sdf)}")
    st.dataframe(sdf.sort_values("liquidated_value_usd_env", ascending=False), width="stretch")


def page_correlation(root: Path, clamp: bool = True):
    st.header("Correlation & Examples")
    df_runs = load_runs(root)
    if df_runs.empty:
        st.info("No runs found.")
        return
    sym_map = load_symbol_map()
    assets = sorted(df_runs["asset"].astype(str).str.lower().unique().tolist())
    assets = _filter_assets(assets)
    asset = st.selectbox(
        "Shock asset",
        assets,
        key="corr_asset",
        format_func=lambda a: sym_map.get(str(a).lower(), str(a)),
    )
    shock = st.slider("Shock magnitude (negative)", -0.50, -0.01, value=-0.10, step=0.025)
    corr = load_corr_matrix()
    prices = load_prices_wide()
    if corr.empty and prices.empty:
        st.warning("Missing correlation inputs (data/prices/corr_365d.csv or historical_prices_365d.csv).")
        return
    if not prices.empty:
        rets = prices.pct_change(fill_method=None).dropna(how="all")
        sigma = rets.std(skipna=True)
        sigma.index = sigma.index.map(str)
    else:
        sigma = pd.Series(dtype=float)
    frozen = {"USDC","USDT","DAI","USDe","sUSDe","USDS","sUSDS","RLUSD","USDtb","TBILL"}
    def _clamp(x: float, lo: float, hi: float) -> float:
        return hi if x > hi else (lo if x < lo else x)
    aset = str(asset).lower()
    if not corr.empty and aset in corr.index:
        rho_row = corr.loc[aset]
    elif not prices.empty and aset in prices.columns:
        rho_row = rets.corrwith(rets[aset]).fillna(0.0)
    else:
        st.warning("Shock asset not present in correlation inputs.")
        return
    sig_s = float(sigma.get(aset, 0.0) or 0.0)
    # Correlation snippet: curated list, show beta not rho
    st.subheader("Correlation snippet (β = ρ·σi/σs)")
    rel = _relevant_assets()
    rows_beta = []
    for addr, label in rel:
        key = str(addr).lower()
        if key not in rho_row.index:
            continue
        rho = float(rho_row.get(key, 0.0) or 0.0)
        sig_i = float(sigma.get(key, 0.0) or 0.0)
        beta = rho * (sig_i / sig_s) if sig_s > 0 and sig_i > 0 else 0.0
        rows_beta.append({"asset": label, "beta": beta})
    beta_df = pd.DataFrame(rows_beta)
    st.dataframe(beta_df, width="stretch")

    # Example lists for common shocks
    st.subheader("Examples (implied moves via β = ρ·σi/σs)")
    default_examples = [-0.05, -0.20, 0.10]
    magnitudes = st.multiselect("Shock magnitudes", default_examples, default=default_examples)
    if not magnitudes:
        magnitudes = default_examples

    # Precompute betas per asset
    betas = {}
    for a in rho_row.index:
        addr = str(a).lower()
        if addr == aset:
            betas[addr] = 1.0
            continue
        rho = float(rho_row.get(addr, 0.0) or 0.0)
        sig_i = float(sigma.get(addr, 0.0) or 0.0)
        if sig_s > 0 and sig_i > 0:
            betas[addr] = rho * (sig_i / sig_s)
        else:
            betas[addr] = 0.0

    for mag in magnitudes:
        title_sym = sym_map.get(aset, aset)
        st.markdown(f"**{title_sym} {mag:+.0%}:**")
        lines = []
        # Include shock asset first (symbol only)
        lines.append(f"  {title_sym}: {mag:+.2%}")
        # Show curated assets in fixed order
        for addr, label in rel:
            key = str(addr).lower()
            if key == aset:
                continue
            move = float(betas.get(key, 0.0)) * mag
            if clamp:
                move = _clamp(move, -0.5, 0.5)
            lines.append(f"  {label}: {move:+.2%}")
        st.code("\n".join(lines))


def page_what_if(root: Path, shock_step: float):
    st.header("What-if (single parameter change)")
    # Expect results under results/what_if/<family>/<level>/dashboard/what_if_*.csv
    wi_root = Path("results/what_if")
    if not wi_root.exists():
        st.info(
            "No what-if outputs found. Run: uv run python v0.0_tests/MVP/run_what_if.py --family bonus --levels 0.05 0.08 0.12"
        )
        return
    families = sorted([p.name for p in wi_root.iterdir() if p.is_dir()])
    if not families:
        st.info("No what-if families present.")
        return
    family = st.selectbox("Family", families)
    levels_dir = wi_root / family
    levels = sorted([p.name for p in levels_dir.iterdir() if p.is_dir()])
    if not levels:
        st.info("No levels present for selected family.")
        return
    level = st.selectbox("Level", levels, format_func=_pretty_level)
    dash_dir = levels_dir / level / "dashboard"
    runs_p = dash_dir / "what_if_runs.csv"
    delta_p = dash_dir / "what_if_deltas.csv"
    if not runs_p.exists() or not delta_p.exists():
        st.warning(f"Missing CSVs for {family}/{level}.")
        return
    df_wi = pd.read_csv(runs_p)
    df_wi["shock"] = pd.to_numeric(df_wi["shock"], errors="coerce")
    df_runs = load_runs(root)
    if df_runs.empty or df_wi.empty:
        st.info("No data to visualize.")
        return
    sym_map = load_symbol_map()
    assets = sorted(
        set(df_runs["asset"].astype(str).str.lower().unique().tolist())
        & set(df_wi["asset"].astype(str).str.lower().unique().tolist())
    )
    assets = _filter_assets(assets)
    if not assets:
        st.info("No overlapping assets between baseline and what-if.")
        return
    asset = st.selectbox(
        "Asset",
        assets,
        key="whatif_asset",
        format_func=lambda a: sym_map.get(str(a).lower(), str(a)),
    )
    # Overlay baseline vs what-if (Agent) for selected asset
    base = df_runs[(df_runs["asset"] == asset) & (df_runs["mode"] == "Agent")].copy()
    alt = df_wi[(df_wi["asset"] == asset)].copy()
    base = _filter_shocks_df(base, shock_step)
    alt = _filter_shocks_df(alt, shock_step)
    col1, col2 = st.columns(2)
    with col1:
        base["series"] = "Baseline"
        alt["series"] = f"{family}:{_pretty_level(level)}"
        plot_df = pd.concat(
            [
                base[["shock", "total_liquidated_value_usd", "series"]],
                alt[["shock", "total_liquidated_value_usd"]].assign(
                    series=f"{family}:{_pretty_level(level)}"
                ),
            ],
            ignore_index=True,
        )
        fig = px.line(
            plot_df,
            x="shock",
            y="total_liquidated_value_usd",
            color="series",
            title="Total Liquidated USD – Baseline vs What-if (Agent)",
        )
        st.plotly_chart(fig, width="stretch")
        if not plot_df.empty:
            fig.update_xaxes(tickvals=sorted(plot_df["shock"].unique()))
    with col2:
        plot_df2 = pd.concat(
            [
                base[["shock", "final_bad_debt_usd"]].assign(series="Baseline"),
                alt[["shock", "final_bad_debt_usd"]].assign(
                    series=f"{family}:{_pretty_level(level)}"
                ),
            ],
            ignore_index=True,
        )
        fig2 = px.line(
            plot_df2,
            x="shock",
            y="final_bad_debt_usd",
            color="series",
            title="Bad Debt – Baseline vs What-if (Agent)",
        )
        st.plotly_chart(fig2, width="stretch")
        if not plot_df2.empty:
            fig2.update_xaxes(tickvals=sorted(plot_df2["shock"].unique()))
    # Deltas table
    df_delta = pd.read_csv(delta_p)
    df_delta["asset"] = df_delta["asset"].astype(str).str.lower()
    df_delta = df_delta[df_delta["asset"] == asset].copy()
    # Show symbol
    if not df_delta.empty:
        df_delta["asset"] = df_delta["asset"].map(lambda a: sym_map.get(a, a))
    st.subheader("Deltas vs Baseline (Agent)")
    st.dataframe(df_delta.sort_values("shock"), width="stretch")


def page_methodology():
    st.header("Methodology")
    st.markdown(
        """
- Propagation: beta-based ΔPi/Pi = βi|S · ΔPS/PS; stables frozen; moves clamped ±50% (guardrailed runs).
- Basis: UB values use environment prices; Agent uses oracle prices post-sync.
- Liquidation: value-conserving; repay ≤ seized value after discount; bad debt socialized only on true shortfall.
- EV: data-driven shock probabilities (historical negative returns). Default vector used per asset if historical tails insufficient.
        """
    )


def main():
    st.set_page_config(page_title="Euler Beta Stress – MVP", layout="wide")
    st.sidebar.header("Run Selection")
    # Prefer a single, finest-grid source so the shock-step toggle drives filtering.
    # Priority: fine_0025 > fine_005 > full; fall back to discovered if none exist.
    preferred = [
        "results/beta_suite_fine_0025",
        "results/beta_suite_fine_005",
        "results/beta_suite_full",
    ]
    selected_str = None
    for p in preferred:
        if (Path(p) / "dashboard" / "runs.csv").exists():
            selected_str = p
            break
    if selected_str is None:
        discovered = discover_results_roots()
        selected_str = discovered[0] if discovered else "results/beta_suite_full"
    root = Path(selected_str)
    st.sidebar.caption(f"Results source: {root} (shock step filters this source)")

    # Shock step selector
    step_label_to_val = {"0.10": 0.10, "0.05": 0.05, "0.025": 0.025}
    step_label = st.sidebar.selectbox("Shock step", list(step_label_to_val.keys()), index=0)
    shock_step = step_label_to_val[step_label]

    tabs = st.tabs(["Overview", "Asset Details", "Breakdown (UB)", "Breakdown (Agent)", "What-if", "Correlation & Propagation", "Methodology"])
    with tabs[0]:
        page_overview(root, shock_step)
    with tabs[1]:
        page_asset_details(root, shock_step)
    with tabs[2]:
        page_breakdown_ub(root)
    with tabs[3]:
        page_breakdown_agent(root)
    with tabs[4]:
        page_what_if(root, shock_step)
    with tabs[5]:
        page_correlation(root, clamp=True)
    with tabs[6]:
        page_methodology()


if __name__ == "__main__":
    main()
