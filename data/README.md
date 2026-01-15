# PigGTEx Data

Data resources for the porcine developmental stage classification pipeline.

## Data Source

- **PigGTEx Portal**: https://piggtex.farmgtex.org/
- **NCBI GEO**: [GSE257558](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE257558)
- **Publications**: [Teng et al. (2024)](https://doi.org/10.1038/s41467-024-47539-8), [Chen et al. (2025)](https://doi.org/)

## Dataset Overview

- **Total samples**: 9,530 RNA-seq samples in PigGTEx v0 MetaTable
- **Samples used for ML**: 1,924 (5 tissues with sufficient developmental stage representation)
- **Key fields**: Tissue, Sex, Age, sequencing/QC metrics

## Developmental Stage Definitions

| Stage | Age Range | Physiological Milestone |
|-------|-----------|------------------------|
| Infant | 0–20 days | Before weaning |
| Early childhood | 21–59 days | Post-weaning growth |
| Pre-pubertal | 60–149 days | Rapid growth phase |
| Post-pubertal | 150–365 days | Sexual maturity |
| Adult | >365 days | Reproductive maturity |

## Tissues Selected for ML Analysis

The following 5 tissues were selected based on sample availability and developmental stage representation:

| Tissue | Total Samples | Samples with Age | Used in ML |
|--------|---------------|------------------|------------|
| Muscle | 1,463 | 914 | ✓ |
| Liver | 607 | 329 | ✓ |
| Brain | 490 | 250 | ✓ |
| Blood | 904 | 284 | ✓ |
| Lung | 170 | 147 | ✓ |

## Complete Tissue Distribution

### Age Distribution by Tissue

|Tissue|Total|Infant (0-20d)|Early childhood (21-59d)|Pre-pubertal (60-149d)|Post-pubertal (150-365d)|Adult (>365d)|Age missing|
|---|---|---|---|---|---|---|---|
|Muscle|1463|272|92|122|423|5|549|
|Unknown|1260|290|53|98|69|37|713|
|Embryo|1050|693|21|0|8|0|328|
|Blood|904|2|211|30|39|2|620|
|Liver|607|95|61|85|55|33|278|
|Brain|490|59|116|19|54|2|240|
|Other|454|80|24|52|45|3|250|
|Small intestine|398|17|110|18|21|14|218|
|Adipose|317|35|0|28|75|6|173|
|Uterus|277|0|0|12|34|6|225|
|Ovary|262|48|0|18|58|6|132|
|Testis|227|19|8|10|29|3|158|
|Heart|219|16|0|22|27|0|154|
|Lung|170|40|26|48|33|0|23|
|pEPSC|139|139|0|0|0|0|0|
|IPEC|127|6|3|0|0|0|118|
|Large intestine|120|0|27|2|9|0|82|
|Spleen|104|29|16|20|14|6|19|
|Oocyte|101|13|0|2|2|0|84|
|Milk|98|0|0|0|0|1|97|
|Synovial membrane|96|0|0|18|0|0|78|
|Bone marrow|86|2|12|0|0|0|72|
|Placenta|86|0|2|0|33|6|45|
|Lymph node|85|5|12|4|3|0|61|
|Artery|73|0|0|0|30|0|43|
|Cartilage|68|6|3|0|14|0|45|
|Kidney|60|10|0|11|10|14|15|
|Skin|53|18|2|0|17|6|10|
|Fetal thymus|48|0|0|0|0|0|48|
|iPS|47|0|0|0|0|0|47|
|Fibroblast|40|0|8|0|0|0|32|
|Stomach|1|0|0|0|1|0|0|

### Sex Distribution by Tissue

|Tissue|Total|Female|Male|Other/pooled|Unknown|
|---|---|---|---|---|---|
|Muscle|1463|364|424|127|548|
|Unknown|1260|256|279|10|715|
|Embryo|1050|101|85|2|862|
|Blood|904|268|186|4|446|
|Liver|607|163|180|6|258|
|Brain|490|162|139|0|189|
|Other|454|48|92|3|311|
|Small intestine|398|109|96|12|181|
|Adipose|317|98|126|0|93|
|Uterus|277|115|22|0|140|
|Ovary|262|174|0|0|88|
|Testis|227|0|96|0|131|
|Heart|219|2|31|0|186|
|Lung|170|29|73|0|68|
|pEPSC|139|0|139|0|0|
|IPEC|127|0|6|0|121|
|Large intestine|120|44|0|1|75|
|Spleen|104|3|25|2|74|
|Oocyte|101|21|0|0|80|
|Milk|98|98|0|0|0|
|Synovial membrane|96|39|21|0|36|
|Bone marrow|86|41|14|0|31|
|Placenta|86|47|33|1|5|
|Lymph node|85|0|17|0|68|
|Artery|73|0|30|0|43|
|Cartilage|68|36|25|0|7|
|Kidney|60|15|8|0|37|
|Skin|53|2|4|1|46|
|Fetal thymus|48|0|0|0|48|
|iPS|47|3|4|0|40|
|Fibroblast|40|8|2|0|30|
|Stomach|1|0|0|0|1|

## Cross-Species Validation Data

Human skeletal muscle data for cross-species comparison:
- **Source**: Schaiter et al. (2024) *Scientific Reports*
- **Location**: `data/human_muscle/`
- **Samples**: 4 infants (7–28 months), 7 adults (30–56 years)