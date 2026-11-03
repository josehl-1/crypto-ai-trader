"""Reglas de decisión compartidas por el scanner, el bot y el backtest."""


def decide(prob_up: float, q50: float, tradable: bool, scfg: dict) -> str:
    if prob_up <= scfg["exit_prob"]:
        return "EXIT"
    if tradable and prob_up >= scfg["buy_prob"] and q50 >= scfg["min_expected_return"]:
        return "BUY"
    return "HOLD"


def score(prob_up: float, q10: float, q50: float, q90: float) -> float:
    """Ranking: convicción direccional + retorno esperado por unidad de incertidumbre."""
    width = max(q90 - q10, 1e-6)
    return round((prob_up - 0.5) * 2 + q50 / width, 4)
