Dear Peter and Ilyas,

Thank you both for the valuable feedback on our manuscript. I want to address each of your points carefully below.

I should note at the outset that the manuscript has been substantially revised since the draft you reviewed, and a number of your excellent suggestions are already incorporated in the current version. I'll flag these below so you can see how they've been addressed.


1. PETER'S FEEDBACK -- CROSS-SPECIES GENERALISABILITY AND MANUSCRIPT QUALITY

Peter, thank you for the kind words about the manuscript and for the thoughtful suggestion regarding cross-species generalisability. Your guidance throughout this project -- from shaping the study's direction to ensuring the manuscript meets the highest standards -- has been instrumental.

You suggested demonstrating how this approach can be broadened to include other species, or showing how the work represents a validated approach through comparability with existing frameworks. This is an excellent point, and we have addressed it directly in the Discussion. Specifically, we include a dedicated paragraph (Discussion, paragraph 5) arguing that our framework is designed to be species-agnostic -- the only required inputs are bulk RNA-seq profiles with age metadata across developmental stages. We draw an explicit parallel with epigenetic clocks (Horvath, 2013; Bi et al., 2017; Schachtschneider et al., 2021), which have successfully extended biological age prediction from humans to mice and pigs using DNA methylation. While those approaches use methylation-based regression, our method applies ordinal machine learning classification to transcriptomic data; the underlying principle -- extending biological age frameworks across species -- is shared. We also note that the growing availability of multi-species developmental transcriptomic resources (Cardoso-Moreira et al., 2019; Coorens et al., 2025) provides the data foundation for applying our approach to additional livestock and model species. This cross-species generalisability framing is a direct result of your insight.

Also, the email address has been corrected to TheobaldPS@Cardiff.ac.uk throughout the manuscript.


2. ABSTRACT VALUES (Ilyas)

The numbers in your suggested abstract appear to reference an earlier draft. The current results, based on the final pipeline with nested cross-validation and hyperparameter tuning, are:

  - Mean balanced accuracy: 0.895 (reported as 0.90 in the abstract)
  - Pearson correlation (cross-species): r = 0.693 (reported as 0.69), 95% CI: 0.54-0.80
  - Directional concordance: 97.0% (64 of 66 genes)
  - Conserved developmental genes: 66 (FDR < 0.10, BH-corrected; |log2FC| > 0.5 in both species)

These values are consistent throughout the manuscript (Abstract, Results, and Discussion). The abstract also reports the bootstrap confidence interval for the cross-species correlation.

Regarding batch effects: because the PigGTEx dataset aggregates RNA-seq profiles from multiple sequencing platforms and library protocols, we performed a within-project fold-change validation to directly test whether cross-project technical variation confounds developmental signals. We computed gene-level fold-changes (Infant vs. Post-pubertal) within the two largest single-project muscle cohorts (PRJNA488311, n = 81; PRJNA486202, n = 75), each spanning four consecutive developmental stages under uniform protocols. Within-project fold-changes correlated strongly with full-dataset estimates (pooled Pearson r = 0.80, 95% bootstrap CI: 0.79-0.81), and the 66 cross-species validated genes showed near-perfect agreement (r = 0.98). This confirms that the developmental signatures reflect genuine biology rather than batch artifacts. These results are reported in the Discussion (limitations paragraph) with full details in Methods and Supplementary Table S5.


3. GENE-BY-GENE RESPONSES (Ilyas)

SORCS2 -- We Agree

We are in full agreement here. SORCS2 (pig FC = -4.67, human FC = -0.95) shows clear postnatal downregulation in both species, consistent with its role in neuromuscular circuit maturation and motor neuron innervation (Thomasen et al., 2023, PMID: 37897724). This gene is discussed as part of Module 1 (Neuromuscular Maturation and Growth Signalling) in the current manuscript, and it is among the 66 conserved developmental genes. Your characterization of its developmental trajectory is entirely consistent with our data.

ACHE -- An Excellent Observation About Bulk RNA-seq Biology

This is a really important point, and I appreciate you raising it. You noted that AChE "rises after birth and peaks in young adults," which reflects the well-established biology of AChE protein concentration at the neuromuscular junction (NMJ). This is absolutely correct at the NMJ level.

However, our bulk RNA-seq data show the opposite trajectory for ACHE mRNA at the whole-tissue level: expression is highest in the Infant stage (mean TPM = 17.5) and progressively declines to baseline in Post-pubertal stages (TPM = 1.3; pig FC = -3.34). This apparent discrepancy is actually well explained in the literature. During postnatal maturation, ACHE mRNA expression becomes increasingly restricted to subsynaptic nuclei at the NMJ rather than being broadly distributed across myonuclei (Boudreau-Lariviere et al., 2000, PMID: 10820184; see also Bhatt et al., PMC6774179). As the muscle matures, fewer nuclei express ACHE at high levels, and expression becomes concentrated at the synapse. In bulk RNA-seq, which averages signal across all nuclei in the tissue, this spatial restriction manifests as a decline in total ACHE mRNA, even though the protein product may be maintained or even enriched locally at the NMJ.

This is a beautiful example of how bulk transcriptomics and protein-level or cell-type-level biology can tell complementary stories. Our data and your biological intuition are both correct -- they simply reflect different levels of resolution. The human data show a concordant decline (human FC = -0.53), confirming that this is a conserved feature of postnatal muscle maturation at the bulk transcriptomic level.

KREMEN1 -- Bulk Tissue Expression Reflects Progenitor Activity

Your suggestion that KREMEN1 postnatal expression gradually rises is an interesting hypothesis, but our data show a postnatal decline (pig FC = -1.69, TPM 30.3 in Infant declining to 9.4 in adults; human FC = -1.43). This trajectory is consistent with KREMEN1's established role as a Wnt antagonist that promotes progenitor cell differentiation (Nakamura et al., 2008, PMID: 18088386; Osada et al., 2006, PMID: 17162372). In infants, higher KREMEN1 expression likely reflects the active Wnt-modulated progenitor activity occurring during early postnatal myogenesis. As the progenitor pool diminishes with maturation and the tissue transitions to homeostatic maintenance, KREMEN1 expression wanes accordingly. The concordant decline in humans supports this interpretation. We discuss KREMEN1 in Module 1 as part of the neuromuscular maturation program.

FGFRL1 -- Confirming Negative Fold Change

You raised a question about the positive fold change for FGFRL1. We can confirm that the fold change is in fact negative (pig FC = -3.03, human FC = -2.10), consistent with its established role in slow muscle fibre specification during embryogenesis (Amann et al., 2014, PMID: 25172430). Expression peaks during the Infant stage (TPM = 77.4) and drops sharply in Early childhood (TPM = 15.6), which is exactly what one would expect as the developmental program for slow fibre specification completes. This gene is discussed in Module 2 (Structural Refinement and ECM Remodelling) in the current manuscript.

IGSF1 and MSS51 -- Not Among the 66 Conserved Genes

Thank you for raising these. We note that neither IGSF1 nor MSS51 is among the 66 cross-species conserved developmental genes identified in the current analysis (which requires FDR < 0.10 and |log2FC| > 0.5 in both species). For reference:

  - IGSF1: Our porcine data show a strong postnatal decline (FC = -5.66), rather than consistent expression. IGSF1 encodes an immunoglobulin superfamily glycoprotein, and its developmental downregulation is consistent with the broader pattern of growth-associated genes declining after the infant period.
  - MSS51: Our data show a positive fold change (FC = +2.20), indicating postnatal upregulation. This is consistent with MSS51's established role as a fast-twitch glycolytic muscle gene (Moyer & Wagner, 2015, PMID: 26634192); its postnatal increase reflects the maturation of fast fibre identity.

Since these genes did not meet the stringent cross-species criteria, they are not discussed in the manuscript, but I'm happy to share the full expression data if it would be helpful for your review.

PDLIM1 and PDLIM3

  - PDLIM1: This gene is not present in our expression dataset, so we cannot comment on its trajectory.
  - PDLIM3: We agree with your characterization of PDLIM3 as a structural Z-disc protein rather than a metabolic gene. Its porcine biology has been well described (Xue et al., 2014, PMID: 24462755). PDLIM3 does not pass our cross-species conservation criteria and is therefore not included in the 66-gene analysis.


4. FIGURE AND DISCUSSION ITEMS (Ilyas)

I'm pleased to report that all of the figure and discussion points you raised are already addressed in the current version of the manuscript:

  - "Porcine" in figure captions (Figs 1-3): All figure captions now explicitly reference porcine data to avoid any ambiguity about the species.
  - Stage boundaries in Fig 4 caption: The Fig 4 caption includes the full stage definitions with age ranges and physiological justifications.
  - DMD discussion: The current manuscript includes a dedicated paragraph in the Discussion (Section 4) discussing Duchenne Muscular Dystrophy applications, including the developmental window mapping between human (2-5 years onset) and porcine stages (Infant through Post-pubertal), and specific molecular markers (TRIM54, FGFRL1) that could serve as therapeutic benchmarks.
  - 5-stage limitation: This limitation is acknowledged in two places: (1) at the end of the Results section, noting that finer-grained cross-species alignment would require human data with greater age resolution, and (2) in the Discussion, explicitly noting the limitations of the human dataset (n = 11; infants and adults only) and the gap covering adolescence and puberty.


Peter, Ilyas -- I genuinely appreciate the depth of your engagement with this work. Peter, your supervision and strategic guidance throughout this project -- from the initial conception of the study through to the final manuscript -- has been essential in ensuring the work is both rigorous and clearly motivated. Ilyas, your expertise in muscle biology and developmental physiology has been invaluable in sharpening our thinking, particularly around the distinction between tissue-level transcriptomics and cell-type-specific protein biology. These are exactly the kinds of nuances that make the difference between a good paper and a rigorous one.

I'm very happy to discuss any of these points further, share additional data, or arrange a call if that would be helpful. Please don't hesitate to reach out.

With warm regards,
Tianyuan
