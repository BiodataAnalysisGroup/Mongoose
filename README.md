# Mongoose

## Abstract

<div align='justify'> Perturbation modelling in single-cell data is crucial for studying molecular changes elicited due to molecular knockouts, chemical compounds, and biological stimulants across health and disease phenotypes. Perturbation modelling is confounded by scarce biological explainability, statistical uncertainty in Deep Learning (DL) predictions in extreme perturbation scenarios, hyperparameter optimization, and limited scalability to multi-omic single-cell data. We present the Mongoose project (Multi-Objective Network Generator Of Optimized Single-cell Experiments) to explore enhancing perturbation modelling in complex single-cell datasets through Multi-Task Learning (MTL). Mongoose combines (i) UnitedNet, a DL framework that employs MTL to simultaneously perform joint group identification and cross-modal prediction with Sharpley values (SHAP) as an explainability component and (ii) perturbation modelling tools like SCING and GenKI, which reconstruct and perturb cell type-specific gene-regulatory networks (GRNs). UnitedNet contains an encoder-decoder-discriminator structure which approximates the statistical characteristics of each modality without prior assumptions. Hence, we claim that UnitedNet can facilitate biologically informed decisions on conducting ensuing digital KOs across reverse-engineered GRNs by providing (i) better cell-type clusters and (ii) SHAP values of significant cross-modal/cell-type associations from complex single-cell and spatial omic datasets. We will showcase the Mongoose approach on complex multi-omic mRNA/protein datasets like the Perturb-CITE-seq (CRISPR knock-outs) and the spatial DBiT-seq mouse embryo dataset, where mRNA-protein associations and spatial niche identification are expected to play pivotal roles in perturbations like in-silico GRN digital KOs. We anticipate that Mongoose will provide perturbational insights closer to ground truths, ultimately highlighting critical transcription factors and signalling pathways with potential translational value. </div>



<br><br>
![ECCB2024](https://raw.githubusercontent.com/BiodataAnalysisGroup/Mongoose/main/Images/ECCB2024.png)

## Execution

To run Mongoose framework for the 3 genes Six2, Gpc3 and Dicer1 follow these steps:

* **Step1:**
<br><br>
Run script  [KO_GENE_OF_INTEREST]_Mongoose_project.ipynb e.g **Six2_Mongoose_project.ipynb**
This script contains GenKI analysis of KO gene. After follows the UnitedNet analysis of
KO gene. At last there is STRINGdb analysis for pathway enrichment based on the responsive
genes found from the 3 methods (GenKI, UnitedNet, Mongoose). Repeat the process for **Gpc3_Mongoose_project.ipynb**
and **Dicer1_Mongoose_project.ipynb**

    From Step1 you will get 3 output files called *genki_filtered_df.csv*, *unit_filtered_df.csv* and *mong_filtered_df.csv*
for **each gene**. 

* **Step2:**
<br><br>
At this point if you run scripts **alluvial_Six2.ipynb**, **alluvial_Gpc3.ipynb** and **alluvial_Dicer1.ipynb**
you will produce the alluvial plot for each gene.

* **Step3:**
<br><br>
For illustration of the pathways found by the enrichment analysis from the digital KO genes Six2, Gpc3 and Dicer1 a heatmap
will be produced. Before this heatmap, running the script **Heatmap_preprocessing.ipynb** for **each gene**. This will
output an organised matrix containing in rows the pathways associated with digital KO genes after the STRINGdb analysis and
in columns the model that found each of the pathways. Value 1 shows that a model found a pathway, value 0 means the opposite.

* **Step4:**
<br><br>
As a final step, run the script **Correlation_heatmaps.ipynb** to plot the final heatmap with all the associations from 
niche modalities of all responsive genes produced in the whole Mongoose project after digit KOs of Six2, Gpc3 and Dicer1.