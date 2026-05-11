# BMC Genomics — Reviewer Comments (Round 1)

**Manuscript:** "A Multi-Tissue Transcriptomic Atlas of Porcine Development Identifies Conserved Molecular Programs with Humans"
**Submission ID:** `e0f004a9-2cfd-4839-8061-e650d2bb8143`
**Decision:** Action needed — revisions required
**Recommended submission deadline:** 2026-05-13
**Notification date received:** 2026-05-11 (this revision cycle)
**Editorial contact:** Sneha Kanade (Editor), Paul Harrison (handling editor) — bmcgenomics@biomedcentral.com
**Corresponding author email used by editor:** TheobaldPS@Cardiff.ac.uk

---

## Editor Comments (Paul Harrison)

> There are several methodological issues that the reviewers point out.

Invitation summary:
> Please ensure the results are accurately reported, any overstated conclusions are rewritten and the limitations of the work fully explained.

Submission must include: revised manuscript file (no tracked changes), point-by-point response (PDF), and optionally a marked-up version uploaded as a related file.

---

## Reviewer 1

In this article, the authors provide a valuable standardized molecular framework for porcine developmental biology. The integration of 1,924 RNA-seq profiles across five tissues and the development of an ordinal machine learning framework represent a meaningful methodological contribution. The cross-species validation with human skeletal muscle data further strengthens the translational relevance of the findings. This is a meaningful study for future nutrition, growth physiology, disease and agriculture studies. However, there are several issues that should be addressed before publication in BMC Genomics.

**R1.1** — For Result 2.1 *A comprehensive transcriptomic atlas of porcine development*. The authors also described "The dataset comprises 1,924 samples spanning five major tissues (brain, liver, muscle, lung, and blood) and covering the entire postnatal lifespan". But, as we all known, the development of any mammals should include prenatal and postnatal stages. However, the authors only collected bulk RNA-seq data from porcine postnatal stages. So, the authors should explain their limitations.

**R1.2** — For Result 2.4 *Evolutionary conservation with human development*. The authors only performed a cross-species analysis using the skeletal muscles between human and pig. However, a previous study (DOI: 10.1038/s41586-019-1338-5) has generated bulk RNA-seq data of brain, liver, and lung in human, macaque, mouse, rat, rabbit, opossum and chicken across their developmental time points from early organogenesis to adulthood.

---

## Reviewer 2

The study presents a comprehensive, machine learning-based framework to establish a standardized molecular definition of developmental stages in pigs, leveraging the extensive PigGTEx resource. The research addresses a topic of significant scientific and applied relevance, responding to the long-standing challenge of improving reproducibility and cross-species translation in swine research and biomedical modeling. The article is based on a substantial dataset (1,924 RNA-seq samples) and a sophisticated analytical framework combining ordinal classification, cross-validation, and cross-species evolutionary analysis. It concludes that biological maturity can be inferred with high accuracy from transcriptomes, and identifies core developmental programs in skeletal muscle that are highly conserved with humans. These findings provide a valuable resource and a novel predictive tool for the community. Several issues need to be addressed to improve the manuscript:

**R2.1** — The study defines five developmental stages based on chronological age bins (e.g., infant: 0–20 days). While these bins align with known physiological milestones, substantial heterogeneity likely exists within each stage due to genetic, environmental, or management factors. The manuscript should discuss how this heterogeneity is addressed by the model or acknowledge it as a limitation. For instance, could the model's performance be influenced by uneven sampling across sub-categories (e.g., different breeds, sexes, or rearing conditions)? A sensitivity analysis or discussion of this variability would strengthen the robustness claims.

**R2.2** — The Methods section describes the use of the Frank & Hall reduction. For complete transparency, it would be helpful to specify how the final stage probability is computed from the K-1 binary sub-model predictions (e.g., using the formula `Pr(y=c) = Pr(y > c-1) - Pr(y > c)`). Additionally, a brief note on how the monotone constraints were applied in LightGBM to enforce the ordinal relationship of the cumulative probabilities would be useful for practitioners seeking to implement similar models.

**R2.3** — The interactive web application is a valuable resource. However, the manuscript should clarify the specific input requirements for users. Since the model was trained on 31,908 genes, what are the recommendations for users who might have RNA-seq data with different gene annotations or microarray data? Providing guidelines on data preprocessing or feature mapping in the manuscript or supplementary materials would increase the tool's usability.

**R2.4** — There are a few formatting issues that need attention. Specifically, in the Supplementary Table 3 and 4 legends, some gene symbols appear to have formatting errors (e.g., "Newomuscalar" instead of "Neuromuscular" in the main text). Additionally, please verify the consistency of the gene nomenclature (e.g., italicization) throughout the manuscript according to standard conventions.

**R2.5** — The references are rather outdated and lack timeliness, especially the absence of literature from the past 3 to 5 years.

> Your research is very meaningful, but I think you need to clarify some of the points I made.

---

## Reviewer 3

This study presents a significant contribution to the field of porcine developmental biology by constructing the first multi-tissue transcriptomic atlas of porcine postnatal development. The authors have successfully developed an ordinal machine learning framework that infers developmental stage from gene expression with high accuracy, offering a valuable tool for standardizing experimental cohorts. The cross-species validation identifying 66 conserved developmental genes between pigs and humans provides compelling molecular evidence for the utility of the pig as a preclinical model. The study is well-designed, the dataset is extensive (1,924 RNA-seq profiles), and the conclusions are largely supported by the data. However, to ensure the robustness and reproducibility of the findings, several methodological details need clarification and additional validation is required.

### Major Concerns

**R3.1** — Batch Effect Evaluation: The dataset integrates samples from the PigGTEx resource, which aggregates RNA-seq profiles from multiple sequencing platforms and library protocols. The authors argue that tree-based models are robust to monotonic platform shifts and provide within-project validation for muscle tissue. While this is a valid point, batch effects can be non-linear and potentially confounded with developmental stages, especially in tissues other than muscle. A simple visualization (e.g., UMAP or PCA) of the data colored by project or batch, rather than just by tissue, would strengthen the claim that batch effects do not drive the developmental predictions. Specifically, while the within-project validation for muscle is robust, similar validation or assessment for other tissues is missing. I recommend the authors provide an evaluation of batch effects across the entire dataset to demonstrate that the developmental signals are not artifacts of technical variation.

**R3.2** — Details on Cross-Species Comparison: The cross-species validation is a key strength of the paper, identifying 97% directional concordance in gene trajectories. However, the methodology for this comparison lacks sufficient detail. Specifically, how were orthologous genes between pig and human defined? Was a strict one-to-one orthologue mapping used? Additionally, while the authors used `|log2 FC|` comparisons to avoid direct expression level comparisons, the pre-processing steps for the human dataset need to be more explicitly described to ensure reproducibility. Clarifying these steps is crucial for assessing the validity of the correlation observed (r=0.69).

**R3.3** — Biological Validation of Developmental Stages: The developmental stages are defined based on physiological milestones (infant, early childhood, etc.). The machine learning model predicts these stages with high accuracy. However, this only proves that the stages are transcriptionally distinct, not that the boundaries defined by chronological age correspond precisely to biological maturation switches. To further validate these stage definitions, the authors could analyze the expression of known developmental marker genes or pathways at the transition points between the defined stages. For instance, showing that genes related to weaning or puberty show sharp expression changes around the defined boundaries would provide independent biological support for the staging framework.

**R3.4** — Interpretability of Tissue-Specific Models: The paper highlights that developmental programs are profoundly tissue-specific, with extremely low cross-tissue feature overlap. While the cross-species markers are discussed in detail, the tissue-specific features driving the predictions in the brain, liver, lung, and blood are less explored. Providing the top feature genes for each tissue-specific model and their biological relevance would significantly enhance the utility of the atlas, offering insights into the unique molecular logic of each tissue's development.

### Minor Concerns

**R3.m1** — Performance Metrics: The performance metrics provided are comprehensive. However, the confusion matrices and precision/recall heatmaps could be better integrated into the main text or referred to more explicitly to highlight specific classification challenges (e.g., misclassification between adjacent stages).

**R3.m2** — Limitations: The authors acknowledge the limitations of the human dataset (small sample size, lack of intermediate stages). It would be beneficial to further discuss the implications of the "infant vs. adult" comparison gap. How might the inclusion of childhood/adolescent data affect the observed correlations? Addressing this theoretically would strengthen the discussion.
