# IEEE Research Paper

Three files:

| File | Purpose |
|---|---|
| `paper.tex` | **The real one.** IEEE conference format (`IEEEtran` class). Upload to [Overleaf](https://overleaf.com) together with `paper_body.tex` and compile — IEEEtran is preinstalled there. This is the version formatted for submission. |
| `paper_body.tex` | All the actual content (abstract → references). Shared by both wrappers, so edits here appear in both. |
| `paper_preview.tex` | Local preview wrapper that approximates the IEEE two-column look using base LaTeX only (for machines without IEEEtran installed). Compile with `pdflatex paper_preview.tex` (run twice). |

## Before submitting — outstanding [TBD] items

1. **Results** — Tables I and II and the abstract's final sentence are
   placeholders. Fill from `outputs/<dataset>/metrics.json` and
   `benchmark_report.json` after running the pipeline on the real datasets.
2. **VersaGuardian citation** — the model is referenced via a footnote
   flag, not a citation, because no primary source was available. Locate
   and add it before submission; until then its reported figures
   (~15.28ms latency, ~20min initialization) must not be presented as fact.
3. **Author block** — affiliation/emails are placeholders. Also confirm
   with Dr. Butt whether he should appear as co-author and in what order.
4. **Verify the three added references** — I added citations for STL
   (Cleveland et al., 1990), DMD (Schmid, 2010), and the SWaT dataset
   (Goh et al., 2016) beyond the five in your proposal. These are
   well-known, real publications, but double-check the bibliographic
   details against Google Scholar before submission.
5. **Architecture figure** — Fig. 1 is a placeholder box; insert your
   actual diagram (e.g. exported from the project docs) via
   `\includegraphics`.
