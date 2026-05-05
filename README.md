## StimSymb: Symbolic Execution Engine for Stim

StimSymb is designed to be a symbolic execution engine for Stim. It uses symbolic stabilizer formalism to extract the symbolic state after executing a Clifford circuit.

## Development

Install development dependencies:

```bash
uv sync --dev
```

Run the formatter:

```bash
uv run ruff format
```

## References

- Scott Aaronson and Daniel Gottesman, "Improved simulation of stabilizer circuits",
  Physical Review A 70, 052328 (2004).
  DOI: https://doi.org/10.1103/PhysRevA.70.052328
- Wenxuan Fang et al., `QuantumSE.jl`:
  https://github.com/njuwfang/QuantumSE.jl
- Craig Gidney, "Stim: a fast stabilizer circuit simulator" (2021).
  arXiv: https://arxiv.org/abs/2103.02202

## Citation

If you use `stimsymb`, cite it as:

```bibtex
@software{liu2026stimsymb,
  author = {Yuhao Liu},
  title = {StimSymb: Symbolic Execution Engine for Stim},
  url = {https://github.com/acasta-yhliu/stimsymb},
  version = {0.1.0},
  year = {2026}
}
```
