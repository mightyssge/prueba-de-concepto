import numpy as np
from envgen.qa import qa_connectivity, qa_viability, qa_distributions

def test_instance_connectivity(distmap, pois_xy):
    r = qa_connectivity(pois_xy, distmap)
    assert r["ok"], f"Conectividad fallida: {r}"

def test_energy_feasibility(pois_eval):
    r = qa_viability(pois_eval, min_pct_ok=95.0)
    assert r["ok"], f"Viabilidad energética < 95%: {r}"

def test_sampling_reproducible(run_A_metrics, run_B_metrics):
    assert run_A_metrics == run_B_metrics, "Mismas seeds y parámetros deberían reproducir la instancia."
