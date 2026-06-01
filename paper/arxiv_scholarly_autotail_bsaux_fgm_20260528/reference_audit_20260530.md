# Reference Audit - 2026-05-30

Scope: `references.bib` in `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/`.

## Style Requirement

For JASIST-oriented writing, the target should be APA-style author-year references. In LaTeX, this means `natbib` author-year in-text citations such as `(Bornmann & Mutz, 2015)` or `(Bornmann and Mutz, 2015)` depending on bibliography style, with clickable blue citation links in the PDF. The current manuscript already uses `natbib` author-year citations and blue citation links.

Reference-list formatting should be normalized before serious submission:

- Journal articles: authors, year, title, journal, volume(issue), pages or article number, DOI when available.
- Conference papers: authors, year, title, full proceedings name, pages, DOI or official URL when available.
- arXiv/preprints: authors, year, title, arXiv identifier URL. Prefer replacing with peer-reviewed versions when they exist.
- Books/reports: authors, year, title, publisher/institution, URL or identifier when available.

## High-Level Findings

- No entry looks obviously fabricated after title-level checks against Crossref, local downloaded metadata, ACL/arXiv-style bibliographic records, or known official publication venues.
- Several entries are real but currently weakly formatted. These should be fixed before arXiv/teacher submission if the paper is intended to look publication-grade.
- Not every reference needs a DOI. Many conference papers and arXiv preprints may instead use ACL Anthology, OpenReview, CVF, NeurIPS, ICLR, AAAI, or arXiv URLs. For journal articles, DOI should be included whenever available.

## Items Requiring Correction

| Key | Status | Issue | Required action |
|---|---|---|---|
| `wu2025scaling` | Real | Author list is incomplete: `Wu, Lingfei and others` is wrong/inadequate. Local metadata says authors are Meng-Jia Wu, Gunnar Sivertsen, Lin Zhang, Fan Qi, Yi Zhang. Missing volume/issue/pages. | Replace author list; add volume 76, issue 11, pages 1470--1487, DOI. |
| `fortunato2018science` | Real | Missing DOI and article number/page `eaao0185`. | Add DOI `10.1126/science.aao0185`; add pages/article number. |
| `glanzel2003bibliometrics` | Real but weak source | It is a course handout/booklet, not a journal article. Current `journal = Course Handouts` is not a proper article venue. | Change type to `@book` or `@misc`; add publisher/institution if retained, or replace with a peer-reviewed bibliometrics source. |
| `boyack2014characterizing` | Real | Crossref title query did not retrieve cleanly; likely metadata incomplete. | Add DOI after manual lookup before finalizing. |
| `colavizza2021citation` | Real | Key says 2021 but year is 2020. Missing DOI. | Rename key or leave key harmless; add DOI `10.1371/journal.pone.0230416`. |
| `li-etal-2022-csl` | Real | Parser/title braces made title-level API search unreliable, but ACL URL exists. Missing DOI if available. | Keep ACL URL; optionally add DOI/ACL anthology id. |
| `beltagy2019scibert` | Real | Crossref search failed because title parsed as `SciBERT`; should use ACL/arXiv official metadata. | Add ACL URL or DOI if available. |
| `devlin2019bert` | Real | Crossref search failed because title parsed as `BERT`; missing ACL URL/DOI. | Add ACL URL `https://aclanthology.org/N19-1423/`. |
| `liu2019roberta` | Real | arXiv preprint entry is acceptable but should include `eprint`/URL. | Add arXiv URL/identifier `1907.11692`. |
| `cui2019wwm` | Real | Published IEEE/ACM TASLP version exists in 2021; current arXiv-only 2019 entry is weaker. | Either keep arXiv with URL or replace with published version DOI `10.1109/TASLP.2021.3124365`. |
| `sun2020ernie` | Likely real topic but current metadata questionable | Current title/authors/AAAI pages look suspicious; Crossref matched unrelated `ERNIE`. The commonly cited Baidu ERNIE paper metadata differs by title/authors. | Needs manual replacement with verified official AAAI/ACL/arXiv metadata or remove if not essential. |
| `buda2018systematic` | Real | Entry type should be `@article`, not `@inproceedings`; journal is Neural Networks, volume 106, pages 249--259. | Change to `@article`; add DOI `10.1016/j.neunet.2018.07.011`. |
| `cao2019ldam` | Real | Missing NeurIPS official URL; Crossref not reliable for NeurIPS title. | Add NeurIPS URL. |
| `ren2020balanced` | Real | Missing NeurIPS/OpenReview style URL; Crossref not reliable. | Add official NeurIPS URL. |
| `menon2021logit` | Real | Missing OpenReview URL. | Add `https://openreview.net/forum?id=37nvvqkCo5`. |
| `wei2022creaming` | Needs stronger verification | Current arXiv-only reference may be real but should be verified with arXiv URL before retaining. | Add arXiv URL if verified; otherwise remove/replace. |
| `goodfellow2015explaining` | Real | Missing arXiv/OpenReview URL; ICLR papers often lack DOI. | Add arXiv URL `https://arxiv.org/abs/1412.6572`. |
| `miyato2016adversarial` | Real | arXiv preprint; missing URL/eprint. | Add arXiv URL `https://arxiv.org/abs/1605.07725`. |
| `zhu2020freelb` | Real | Missing OpenReview URL. | Add official ICLR/OpenReview URL. |
| `jiang2020smart` | Real | Missing ACL URL/DOI. | Add ACL URL. |
| `liang2021rdrop` | Real | Missing NeurIPS URL/arXiv. | Add official URL. |
| `dietterich1998approximate` | Real | Missing DOI. | Add DOI `10.1162/089976698300017197`. |

## Entries That Look Acceptable With Minor DOI/URL Improvements

| Key | Status | Notes |
|---|---|---|
| `garfield1972citation` | Real | Add DOI `10.1126/science.178.4060.471`. |
| `narin1976evaluative` | Real report/book | No DOI expected; add publisher/institution details if possible. |
| `bornmann2015growth` | Real | Add DOI `10.1002/asi.23329`. |
| `waltman2012new` | Real | Add DOI `10.1002/asi.22748`. |
| `cohan2018discourse` | Real | Add DOI `10.18653/v1/N18-2097` or ACL URL. |
| `lo2020s2orc` | Real | Add DOI `10.18653/v1/2020.acl-main.447` or ACL URL. |
| `ammar2018construction` | Real | Add DOI `10.18653/v1/N18-3011` or ACL URL. |
| `luan2018multi` | Real | Add DOI `10.18653/v1/D18-1360` or ACL URL. |
| `cui2020revisiting` | Real | Add DOI `10.18653/v1/2020.findings-emnlp.58` or ACL URL. |
| `he2009learning` | Real | Add DOI `10.1109/TKDE.2008.239`. |
| `lin2017focal` | Real | Add DOI `10.1109/ICCV.2017.324`. |
| `zhang2021distribution` | Real | Add DOI `10.1109/CVPR46437.2021.00239`. |

## Recommended Next Action Status

The cleanup actions below have been applied. The remaining citation style choice is intentional: the manuscript keeps `plainnat` for arXiv compatibility while normalizing metadata fields and enabling blue clickable author-year citations through `natbib` and `hyperref`.

## Cleanup Applied - 2026-05-30

The bibliography was cleaned with a conservative dependency rule: cited references were retained unless the corresponding sentence could remain supported by stronger sources; uncited questionable entries were removed from `references.bib` only.

Applied changes:

- Removed `glanzel2003bibliometrics` from the Related Work citation because it was a course handout-style source and the same sentence remains supported by `garfield1972citation` and `narin1976evaluative`.
- Removed uncited BibTeX entries: `colavizza2021citation`, `sun2020ernie`, `zhang2021distribution`, and `wei2022creaming`. These removals do not change the manuscript text because none of the keys appeared in `main.tex`.
- Fixed `wu2025scaling` using local Wiley/JASIST metadata: full author list, volume 76, issue 11, pages 1470--1487, DOI `10.1002/asi.70004`.
- Replaced the incorrect `boyack2014characterizing` metadata with the verified JASIST article `Creation of a Highly Detailed, Dynamic, Global Model and Map of Science`, volume 65, issue 4, pages 670--685, DOI `10.1002/asi.22990`. The correction preserves the original citation key so the manuscript text remains unchanged.
- Added DOI or official URL metadata for retained journal, ACL, arXiv, OpenReview, NeurIPS, and ICCV references where a stable source was available.
- Normalized DOI/URL formatting: entries with DOI now keep DOI only, while entries without DOI keep official URL/catalog links. The LaTeX DOI macro renders DOI references as clickable `https://doi.org/...` links.
- Upgraded `cui2019wwm` from the arXiv record to the related peer-reviewed IEEE/ACM TASLP article using DOI `10.1109/TASLP.2021.3124365`.
- Corrected `buda2018systematic` from `@inproceedings` to `@article` in `Neural Networks`.

Post-cleanup validation:

- `main.tex` cites 29 unique bibliography keys.
- `references.bib` contains exactly 29 entries.
- There are no missing cited keys and no uncited BibTeX entries.
- `tectonic --keep-intermediates main.tex` successfully generated `main.pdf` and `main.bbl`.
- Remaining warnings are non-fatal PDF-version and underfull-line warnings, not citation failures.
