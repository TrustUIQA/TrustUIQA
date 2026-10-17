# Paper (IEEE conference format)

`main.tex` (IEEEtran, conference) + `refs.bib`. Figures go in `figures/`.

## Build on Overleaf
1. Overleaf → **New Project → Upload Project** → upload a zip of this `paper/` folder (or the files one by one).
2. Menu → Compiler: **pdfLaTeX**. IEEEtran is built into Overleaf.
3. Recompile. Missing figures render as labelled boxes, so the paper compiles before the figures exist.

## Add the figures
1. Run Kaggle **notebook 06** (`notebooks/06_kaggle_policies_and_figures.ipynb`) as a commit.
2. Download `paper_results.zip` from that version's **Output** tab.
3. Copy `paper/fig_*.pdf` from the zip into this folder's `figures/`:
   `fig_o1_srcc.pdf`, `fig_o2_sweeps.pdf`, `fig_o3_delta.pdf`, `fig_o5_policies_map.pdf` (and `fig_o5_policies.pdf`).

## Before submission
- Replace the author block (name, affiliation, email).
- Check every `refs.bib` entry marked `% VERIFY` against the publisher page.
- Check the target venue's page limit; the Discussion and Related Work are the easiest to trim.
