# Lung cross-species: data-availability audit

## Why lung is missing from `cross_species_all_tissues.json`
Cardoso-Moreira *et al.* 2019 (E-MTAB-6814) sampled 7 organs — Brain,
Cerebellum, Heart, Kidney, Liver, Ovary, Testis. Lung was not included. This
is the binding constraint that the response letter (R1.2) already
acknowledges.

## Candidate alternative human datasets surveyed

| Source / GEO | Species | Tissue | Stages | Bulk? | Direct download? | Verdict |
|---|---|---|---|---|---|---|
| LungMAP (LGEA portal) | Human | Lung (4 sorted cell pops) | 1d–8y (n=24 donors) | Sorted-cell bulk | Portal only, no GEO matrix | Best biological fit; **needs portal scraping** |
| GSE122960 (Reyfman 2019) | Human | Lung donors + IPF | Adult only | scRNA-seq | Yes (HDF5) | Single-cell; adult only |
| GSE124872 | Mouse | Lung | 3 vs 24 mo | Bulk + scRNA | Yes | Wrong species |
| GSE292089 | Mouse | Lung neurons | n/a | Bulk | Yes | Wrong species |
| GSE81089 | Human | Lung tumour + adjacent | Adult only | Bulk | Yes (FPKM) | Cancer/adult — wrong context |
| GSE119911 | Human | Lung tumour | Adult only | scRNA | Partial | Cancer/adult |
| GSE226572 | Human | PBMC | n/a | scRNA | Yes | Wrong tissue |
| GSE270890 | Human | Monocytes + breast spheroids | n/a | Bulk | Yes | Wrong tissue |
| EBI Expression Atlas | — | — | — | — | Browse only via UI | Not programmable here |
| GTEx adult Lung | Human | Lung | Adult only | Bulk (TPM) | Yes (registration) | No paediatric ages |

## Feasibility verdict
- **No off-the-shelf healthy-human postnatal lung bulk RNA-seq matrix** matching
  the Cardoso-Moreira schema is directly downloadable from GEO/SRA in this
  session.
- The Bandyopadhyay *et al.* 2024 LungMAP dataset (24 donors, 1d–8y) is the
  right biological match but the deposited data are on the LungMAP/LGEA portal
  rather than as a GEO matrix.
- A hybrid approach (GTEx adult lung + a separate paediatric study) would be
  defensible but adds a second batch confound that needs careful handling.

## Recommended next steps (outside this session)
1. Apply for LungMAP / LGEA data access; download the bulk matrices.
2. Map LungMAP age strings (1 day, 1 month, ..., 8 years) to the porcine
   stage scheme (Infant 0–20d, Early childhood 21–59d, Pre-pubertal 60–149d).
3. Re-run `autoresearch/run_purity_generic.py` with `--pig-tissue Lung`
   `--human-tissue Lung` once the data are in place.
4. Note in the manuscript that LungMAP is sorted-cell bulk RNA-seq, whereas
   PigGTEx lung is whole-tissue bulk; the cross-species comparison should be
   restricted to genes with broad expression across cell types or weighted by
   cell-type composition.

## Implication for the response letter
The current response letter (R1.2) already states that lung cross-species is
not feasible with Cardoso-Moreira 2019. After this audit, the language can be
strengthened: "the canonical multi-organ developmental atlas Cardoso-Moreira
2019 has no lung samples; LungMAP and LGEA contain healthy postnatal human
lung bulk RNA-seq but use sorted cell populations rather than whole tissue,
making direct comparison to PigGTEx whole-lung bulk RNA-seq methodologically
non-trivial. We have therefore deferred a clean human-pig lung cross-species
analysis to future work."
