# Early Health Risk Detection Using Self Reported Data
An Ensemble Machine Learning Pipeline with Attention-Based Neural Networks for Early Health Risk Detection Using Self-Reported Data


# An Ensemble Machine Learning Pipeline with Attention-Based Neural Networks for Early Health Risk Detection Using Self-Reported Data

# Abstract

Early identification of individuals at elevated health risk is crucial for preventive interventions. This work presents a modular, end to end machine learning pipeline designed to operate exclusively on self reported, open source data (e.g., NHANES, BRFSS, UCI Heart Disease). The architecture integrates five classical classifiers Gradient Boosting, Random Forest, Logistic Regression, XGBoost, and LightGBM with a hybrid neural network that incorporates a multi head self attention mechanism. To prevent data leakage and ensure realistic performance estimation, a strict four way split (train/validation/test/holdout) is employed, and the holdout set is never touched until final evaluation. Anti overfitting measures include L2 regularization, dropout (40%), batch normalization, early stopping, and sample weighting derived from IsolationForest outlier scores. Extensive feature engineering creates clinically meaningful composites (e.g., cardiovascular risk proxy, metabolic syndrome score, lifestyle index). Applied to the UCI Heart Disease dataset (n=303), the pipeline achieves a holdout accuracy of 0.857 and ROC AUC of 0.941, with a train holdout generalization gap of only 1.8 percentage points, indicating excellent out of sample performance. All code, visualizations, and model metadata are made publicly available to promote reproducibility in computational health research.
<p>Keywords: Ensemble learning; Attention mechanism; Health risk prediction; Self reported data; Open source pipeline; Model generalization</p>

<p>
<img width="1185" height="1032" alt="viz_tsne_holdout" src="https://github.com/user-attachments/assets/f05d24f8-5404-4ba1-8c90-30acc3f8f3f4" />
<img width="2084" height="1475" alt="nn_training_history" src="https://github.com/user-attachments/assets/28bf1506-eabd-4bd1-804e-eb90959a11f5" />
<img width="1784" height="593" alt="feature_pca_variance" src="https://github.com/user-attachments/assets/a164aed1-4fd8-49c9-902a-4d6083b9d2ff" />
<img width="1485" height="1181" alt="feature_importance" src="https://github.com/user-attachments/assets/cee80f69-e7c6-445e-b13f-ebe001e8506c" />
<img width="1036" height="818" alt="eval_roc_curves" src="https://github.com/user-attachments/assets/a3f1a983-d60d-44d4-9fd9-4ff1bf58a7d0" />
<img width="1036" height="818" alt="eval_pr_curves" src="https://github.com/user-attachments/assets/b6776b62-1a27-4421-ae42-3c739c21dcf9" />
<img width="1485" height="1031" alt="eval_perm_importance_holdout" src="https://github.com/user-attachments/assets/e6934437-b55c-491e-9683-2de6283bc832" />
<img width="2066" height="732" alt="eval_model_comparison" src="https://github.com/user-attachments/assets/6e80a476-ed86-4301-82ac-54c0597fd2d2" />
<img width="2084" height="884" alt="eval_metrics_bar" src="https://github.com/user-attachments/assets/7fb83d89-7807-4e07-919e-00e22528e0bf" />
<img width="2382" height="1770" alt="eval_holdout_deep_dive" src="https://github.com/user-attachments/assets/b959db5c-bbce-442e-acca-3811247906ef" />
<img width="2084" height="741" alt="eval_generalization_gap" src="https://github.com/user-attachments/assets/4e1e8435-6946-472d-9faf-601eb87848dc" />
<img width="1184" height="733" alt="eval_cv_results" src="https://github.com/user-attachments/assets/d2611a52-392e-4e0d-8974-afe3164bf861" />
<img width="2945" height="593" alt="eval_confusion_matrices" src="https://github.com/user-attachments/assets/5cf73aba-e6b4-46a4-a9f1-d552a1404b91" />
<img width="1485" height="581" alt="eda_outlier_summary" src="https://github.com/user-attachments/assets/8522719f-2331-415a-9dd7-0d41122ea885" />
<img width="1784" height="732" alt="eda_missing_values" src="https://github.com/user-attachments/assets/eccc0d9c-3e50-4015-8667-3afd8da911ae" />
<img width="2385" height="1551" alt="eda_feature_vs_target" src="https://github.com/user-attachments/assets/789a8c8c-225f-405c-a7ef-fcd096cf9c3a" />
<img width="2385" height="1771" alt="eda_feature_distributions" src="https://github.com/user-attachments/assets/81c9c812-dc43-4671-b347-658fa64191e2" />
<img width="1914" height="1783" alt="eda_correlation_heatmap" src="https://github.com/user-attachments/assets/55ffffdf-a47c-410e-bfea-eb9dac4533fa" />
<img width="1671" height="593" alt="eda_class_distribution" src="https://github.com/user-attachments/assets/4431611f-0090-41b6-b56c-01f1767b42f1" />

</p>
