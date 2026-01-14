# Presentation Speech Notes
## Developmental Stage Classification in Pigs: A Machine Learning Approach

---

## Slide 1: Title Slide
> **"Developmental Stage Classification in Pigs - A Machine Learning Approach Using Transcriptomic Data"**

"Good morning/afternoon everyone. My name is Tianyuan Liu from Cardiff University. Today I'll be presenting our work on classifying developmental stages in pigs using machine learning and transcriptomic data. This project addresses a fundamental challenge in developmental biology: can we use gene expression to accurately determine an animal's developmental stage?"

---

## Slide 2: The Gap in Human Developmental Data

"To understand why this work matters, let's start with the human perspective. This timeline from the developmental GTEx project shows a critical gap in our knowledge. While we have comprehensive transcriptomic data for adult humans, comprehensive human developmental transcriptomics remains limited—particularly during key developmental windows. This is where pigs become invaluable as a model organism. By creating accurate molecular staging tools for pigs, we can bridge this gap and use pigs to study developmental processes that are difficult to capture in humans."

---

## Slide 3: The Challenge

"So what are the current challenges we face?

On the left side, you'll see the gaps in our field. First, there's no molecular standard to stage pigs—we typically rely on chronological age, which is unreliable because animals develop at different rates. Second, tissue signals differ significantly, making cross-tissue staging inconsistent. Third, the alignment between pig and human development is unclear, and shared biomarkers are underutilized.

Our goal—shown on the right—is three-fold:
1. Use gene expression to define molecular developmental stages
2. Align pig stages to the human timeline using conserved biomarkers
3. Create a unified, cross-species molecular clock

Essentially, we want to go from gene expression to pig age, then from pig age to human age."

---

## Slide 4: The PigGTEx Project

"Our data comes from the PigGTEx project, which is part of the FarmGTEx consortium—a major initiative to build comprehensive catalogues of genetic regulatory variants in farm animals.

The project has three key objectives:
- Understanding genetic mechanisms that link genetic variation to gene expression
- Leveraging pigs as biomedical models for human physiology and disease
- Informing sustainable agriculture through better breeding strategies

This is a massive public resource that integrates thousands of RNA-seq samples to map the pig regulatory genome."

---

## Slide 5: Data Quality Challenges

"However, working with real-world data comes with significant challenges.

Looking at missing data: 52% of samples are missing age information, 52% are missing sex labels, and 13% have unknown tissue categories.

The age format inconsistencies present another hurdle. Ages are stored as text in various formats—days, weeks, months, years—or simply 'unknown'. 

Our challenge was to harmonize all of this into a unified numeric format that we could use for machine learning."

---

## Slide 6: Data Processing Pipeline

"Here's how we addressed these challenges through our data processing pipeline.

First, age harmonization: we parsed text formats into days and mapped them to five biological stages—Infant, Early childhood, Pre-pubertal, Post-pubertal, and Adult.

Second, for expression preprocessing, we applied log2 TPM plus 1 transformation to normalize the gene expression values.

Third, quality control: we filtered samples with missing stage information and explicitly encoded unknown sex values.

Fourth—and this is important—we developed adaptive classification schemes. Because different tissues have different sample sizes, we used 4-class classification for tissues with more data like Muscle and Liver; and 2-class binary classification for tissues with fewer samples like Lung, Blood, and Brain. This ensures sufficient samples per class for robust machine learning training.

After this processing, we have 2,467 curated samples across 8 major tissues."

---

## Slide 7: Classification Schemes

"This figure shows the different classification schemes we use across tissues. You can see how we adapt the granularity based on sample availability, ensuring that each class has enough samples for training while maximizing the biological resolution where possible."

---

## Slide 8: Dataset Overview

"Here's an overview of our final dataset. We have 2,467 samples across 8 tissues and cell types—muscle, brain, liver, blood, intestine, lung, adipose, and testis. The bar chart on the right shows the sample distribution across these tissues, with muscle and liver having the most samples."

---

## Slide 9: Methodology Overview

"Now let me walk you through our methodology. We have three main components:

First, classification using multi-resolution schemes—4-class, 3-class, and binary as I mentioned.

Second, machine learning using Ordinal LightGBM with a binary reduction strategy—I'll explain why we chose this approach in the next slides.

Third, validation through tissue-specific performance metrics and cross-tissue robustness testing."

---

## Slide 10: The Machine Learning Problem

"Let me explain the machine learning problem in simple terms.

What is X—our features? Gene expression values from RNA-seq. Each sample has over 20,000 genes with expression values—essentially each sample is one pig's complete gene expression profile.

What is Y—our label? The developmental stage we want to predict—Embryonic, Juvenile, or Adult, depending on the classification scheme.

In simple terms: given a pig's gene activity, which developmental stage is it in?"

---

## Slide 11: Why Machine Learning?

"Why did we choose machine learning, specifically LightGBM?

We considered three approaches:

Traditional regression assumes linear relationships, but genes interact non-linearly. This approach would miss complex patterns.

Deep learning can capture these non-linear patterns, but it needs thousands of samples to train effectively. We only have hundreds per tissue—not enough.

LightGBM is the sweet spot. It handles non-linear patterns using decision trees, but works effectively with small sample sizes. This makes it the best choice for our problem—getting maximum performance with limited sample sizes."

---

## Slide 12: How Does a Decision Tree Work?

"Let me briefly explain how a decision tree works using a simple example with just three genes.

The tree starts at the top and asks yes/no questions about gene expression values. For example: 'Is Gene1 expression greater than 5?' If yes, we classify as Adult. If no, we ask about Gene2, and so on.

As an example, if a sample has Gene1 equal to 6.2, we ask 'Is 6.2 greater than 5?' Yes, so we classify that sample as Adult.

A single tree asks simple questions about genes to classify developmental stage."

---

## Slide 13: From One Tree to Many Trees

"Of course, one tree can make mistakes. That's why we use many trees that vote together.

In this example, we have four trees. Three predict Juvenile and one predicts Adult. The majority vote wins—so our final prediction is Juvenile.

Why multiple trees? Each tree sees the data slightly differently. Individual mistakes cancel out. The majority vote is more reliable than any single tree.

More trees equals more reliable predictions."

---

## Slide 14: What Makes LightGBM Special - Boosting

"What makes LightGBM special is boosting.

In boosting, each new tree learns from the previous trees' mistakes. Tree 1 makes an initial guess—some errors remain. Tree 2 focuses specifically on those errors and fixes Tree 1's mistakes. Tree 3 refines further. And so on.

The final prediction is a weighted sum of all trees.

Think of it as trees working as a team, where each tree specifically targets and fixes the mistakes of its predecessors."

---

## Slide 15: Why Ordinal Classification Matters

"Now, why ordinal classification? This is crucial.

Developmental stages have a natural order: Embryonic comes before Juvenile, which comes before Adult.

Here's the key insight: misclassifying an Adult as Embryonic is worse than misclassifying as Juvenile—because it's further from the true answer on the developmental timeline.

Standard classification treats all classes as equal—there's no concept of distance between classes.

Ordinal classification respects this stage ordering. It penalizes distant errors more heavily and produces better calibrated probabilities. This makes our predictions more biologically meaningful."

---

## Slide 16: Performance Metrics Overview

"Before showing results, let me quickly explain the metrics we use.

Precision asks: of all the samples we predicted as a certain stage, what proportion was actually correct?

Recall asks: of all the actual samples of a stage, what proportion did we correctly identify?

F1 Score is the harmonic mean of precision and recall—balancing both.

MCC—Matthews Correlation Coefficient—is particularly important for imbalanced datasets because it considers all parts of the confusion matrix."

---

## Slide 17: Model Performance

"Here are our model performance results. The figure shows balanced accuracy, F1, precision, recall, and MCC across all tissues.

The key finding is that we achieve strong performance across all metrics. This demonstrates that gene expression can reliably predict developmental stage."

---

## Slide 18: Confusion Matrices

"These confusion matrices show the detailed classification results for each tissue. The diagonal elements represent correct predictions—you can see strong diagonal patterns across most tissues, indicating high accuracy. This confirms our model's ability to distinguish between developmental stages."

---

## Slide 19: Age Correlation

"This figure shows the correlation between predicted developmental stage and actual chronological age. The strong correlations validate that our classifications align with biological expectations—samples classified as later developmental stages do indeed come from older animals."

---

## Slide 20: Feature Overlap Analysis

"Now let's look at biological insights. This is where things get really interesting.

When we analyze feature overlap—which genes are important for classification in each tissue—we find something striking. The Jaccard similarity is only about 0.3%.

This means each tissue uses distinct marker genes to classify developmental stage. But this raises an important question: do these different genes converge on shared biological processes?"

---

## Slide 21: Tissue Specificity - UMAP

"This UMAP visualization shows why tissues use different genes. You can see clear separation between tissues—each tissue has distinct gene expression patterns.

This explains the low gene overlap: tissue-specific gene expression drives unique feature selection. 

But here's the key insight: despite using different genes, developmental trajectories may share common pathways. Different roads can lead to the same destination."

---

## Slide 22: Pathway Enrichment Analysis

"And indeed, that's exactly what we find. This pathway enrichment analysis using KEGG and Reactome databases reveals shared developmental programs across tissues.

Even though tissues use different genes, these genes converge on common biological processes: gene regulation, metabolism, and extracellular matrix organization.

This is a fundamental insight about development: tissues use different molecular players but orchestrate similar developmental programs."

---

## Slide 23: Summary & Future Directions

"Let me summarize our key findings:
- We can accurately stage pigs from transcriptomics data
- We discovered tissue-specific molecular clocks—each tissue uses distinct genes
- But these converge on conserved developmental programs

Looking ahead, we have three next steps:
1. Cross-species integration—mapping pig genes to human orthologs
2. Biomarker validation—validating our findings across species

This future work will be led by Rujing."

---

## Slide 24: Thank You

"Thank you for your attention. I'm happy to take any questions. You can reach me at liut33@cardiff.ac.uk."

---

## General Tips for Delivery

1. **Pace**: Allow about 1-2 minutes per slide for content slides, less for transition/figure slides
2. **Figures**: Point to specific elements when explaining figures
3. **Pauses**: Pause after key findings to let them sink in
4. **Engagement**: Make eye contact, especially during key messages
5. **Questions**: Be prepared for questions about:
   - Sample size justification for ML
   - Why LightGBM over other methods
   - Biological interpretation of pathway findings
   - Plans for cross-species validation
