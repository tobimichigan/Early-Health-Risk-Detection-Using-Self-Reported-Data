# Early Health Risk Detection Using Self Reported-Data
An Ensemble Machine Learning Pipeline with Attention-Based Neural Networks for Early Health Risk Detection Using Self-Reported Data


# An Ensemble Machine Learning Pipeline with Attention-Based Neural Networks for Early Health Risk Detection Using Self-Reported Data

# Abstract

Early identification of individuals at elevated health risk is crucial for preventive interventions. This work presents a modular, end to end machine learning pipeline designed to operate exclusively on self reported, open source data (e.g., NHANES, BRFSS, UCI Heart Disease). The architecture integrates five classical classifiers Gradient Boosting, Random Forest, Logistic Regression, XGBoost, and LightGBM with a hybrid neural network that incorporates a multi head self attention mechanism. To prevent data leakage and ensure realistic performance estimation, a strict four way split (train/validation/test/holdout) is employed, and the holdout set is never touched until final evaluation. Anti overfitting measures include L2 regularization, dropout (40%), batch normalization, early stopping, and sample weighting derived from IsolationForest outlier scores. Extensive feature engineering creates clinically meaningful composites (e.g., cardiovascular risk proxy, metabolic syndrome score, lifestyle index). Applied to the UCI Heart Disease dataset (n=303), the pipeline achieves a holdout accuracy of 0.857 and ROC AUC of 0.941, with a train holdout generalization gap of only 1.8 percentage points, indicating excellent out of sample performance. All code, visualizations, and model metadata are made publicly available to promote reproducibility in computational health research.
<p>Keywords: Ensemble learning; Attention mechanism; Health risk prediction; Self reported data; Open source pipeline; Model generalization</p>

<p><img width="916" height="696" alt="tsne_2d" src="https://github.com/user-attachments/assets/d8ae443d-db41-4012-b4b5-eba9072713e5" />
<img width="1178" height="697" alt="threshold_tuning_val" src="https://github.com/user-attachments/assets/097731b7-ecf4-44de-bb63-3ad6a4187942" />
<img width="1178" height="697" alt="threshold_tuning_test" src="https://github.com/user-attachments/assets/f686f31a-38d4-48b5-8cce-1712ecf44c3e" />
<img width="1178" height="697" alt="threshold_tuning_holdout" src="https://github.com/user-attachments/assets/21c497ed-77e2-4944-b6e0-d5e38c2d9ac6" />
<img width="1058" height="817" alt="roc_val" src="https://github.com/user-attachments/assets/4b8105fe-1137-4541-ba7a-1183b36609b0" />
<img width="1058" height="817" alt="roc_test" src="https://github.com/user-attachments/assets/fbd914df-e2b4-4a93-aedf-d1450b063aee" />
<img width="1058" height="817" alt="roc_holdout" src="https://github.com/user-attachments/assets/afbb54bd-2bad-4c8b-b0f3-485b01a0ab1f" />
<img width="1058" height="817" alt="pr_val" src="https://github.com/user-attachments/assets/1f77db0d-a97b-4d66-a617-27353c538d31" />
<img width="1058" height="817" alt="pr_test" src="https://github.com/user-attachments/assets/6c0a4a89-4b01-4e23-a41b-f10d08e290a7" />
<img width="1058" height="817" alt="pr_holdout" src="https://github.com/user-attachments/assets/ed385cd3-5539-471b-a33c-b34879b23d59" />
<img width="897" height="576" alt="per_class_recall" src="https://github.com/user-attachments/assets/6b07ecca-9ff8-4d18-bb07-baa35cd76143" />
<img width="911" height="696" alt="pca_2d" src="https://github.com/user-attachments/assets/03c8ff12-257a-44a0-ad0e-b8ab8ef16472" />
<img width="938" height="456" alt="outlier_weights" src="https://github.com/user-attachments/assets/5ff42fac-0dab-4c17-97a6-e08ec1f8130f" />
<img width="1180" height="577" alt="model_leaderboard" src="https://github.com/user-attachments/assets/c3a689a7-9ab0-4199-8a97-5241839a5e3a" />
<img width="1661" height="699" alt="metrics_comparison_val" src="https://github.com/user-attachments/assets/a88ac165-3df6-4886-99f2-3b372725d471" />
<img width="1661" height="699" alt="metrics_comparison_test" src="https://github.com/user-attachments/assets/5eeb5928-9d27-4968-a6eb-f196c530f6cc" />
<img width="1661" height="699" alt="metrics_comparison_holdout" src="https://github.com/user-attachments/assets/39c34785-10d2-452f-a139-ff6e31b0ff1d" />
<img width="1313" height="700" alt="holdout_metrics_heatmap" src="https://github.com/user-attachments/assets/87770485-ac0f-4980-8cef-2db47194a029" />
<img width="1418" height="697" alt="generalisation_gap" src="https://github.com/user-attachments/assets/4142207e-2726-4ba2-bd5c-5bf25b3c79ee" />
<img width="1180" height="696" alt="feat_imp_RandomForest" src="https://github.com/user-attachments/assets/b21fa92d-26ab-48e5-a1b0-06e72fa2ec18" />
<img width="1180" height="696" alt="feat_imp_GBM" src="https://github.com/user-attachments/assets/e8f8149d-21a5-4b79-8f90-33f7d655a89e" />
<img width="1418" height="1175" alt="eda_10_qq_plots" src="https://github.com/user-attachments/assets/0a791b62-076a-4b96-b016-5a1d2b44cc63" />
<img width="940" height="457" alt="eda_09_target_correlation" src="https://github.com/user-attachments/assets/7621e2c0-a14e-4fe3-8cc2-b3bd0aee8be9" />
<img width="1421" height="459" alt="eda_08_missing_values" src="https://github.com/user-attachments/assets/df472539-0abe-4f52-961b-ea40116523bb" />
<img width="1305" height="1261" alt="eda_07_pairplot" src="https://github.com/user-attachments/assets/db767964-2a17-4ae2-943e-e07bd8746733" />
<img width="1778" height="1763" alt="eda_06_bivariate_scatter" src="https://github.com/user-attachments/assets/4a0961e8-c967-4e9b-88e8-e77f9df5089a" />
<img width="2858" height="1175" alt="eda_05b_categorical_stacked_percent" src="https://github.com/user-attachments/assets/58951003-5dff-4ca2-a3d7-91aeb3cc739a" />
<img width="2861" height="1175" alt="eda_05_categorical_counts" src="https://github.com/user-attachments/assets/68c987d3-507c-4bf5-a476-4e0fc802bcd8" />
<img width="2378" height="1175" alt="eda_04_box_violin_by_target" src="https://github.com/user-attachments/assets/7691892b-d458-4900-81f9-32aa98a84ec1" />
<img width="1441" height="1180" alt="eda_03_correlation_heatmap" src="https://github.com/user-attachments/assets/7c7f9183-a6d1-475f-ba8e-1e0d5473a7d4" />
<img width="2381" height="943" alt="eda_02_numeric_distributions" src="https://github.com/user-attachments/assets/02241832-b771-497d-bca8-16a444c87508" />
<img width="1333" height="459" alt="eda_01_class_distribution" src="https://github.com/user-attachments/assets/04816440-7da9-48b5-b81d-94349a01e627" />
<img width="2141" height="587" alt="dnn_training_history" src="https://github.com/user-attachments/assets/c6620897-96e3-4534-885d-b29901513144" />
<img width="1058" height="817" alt="cumulative_gains_val" src="https://github.com/user-attachments/assets/744ed5d5-7fe5-41bc-bf52-a9f0d3f26b71" />
<img width="1058" height="817" alt="cumulative_gains_test" src="https://github.com/user-attachments/assets/02dbad69-24de-4485-8206-2494f0bbf3c8" />
<img width="1058" height="817" alt="cumulative_gains_holdout" src="https://github.com/user-attachments/assets/8c54519b-3bb7-44a6-8040-171095877a8c" />
<img width="1178" height="699" alt="cross_validation_boxplot" src="https://github.com/user-attachments/assets/ba95094d-9c33-437a-a8fb-e4c343e4ab61" />
<img width="668" height="576" alt="cm_VotingEnsemble_val" src="https://github.com/user-attachments/assets/44a94f8f-58ff-42a4-9e9c-a648cfdc046a" />
<img width="668" height="576" alt="cm_VotingEnsemble_holdout" src="https://github.com/user-attachments/assets/887f44c7-bbaf-4cb2-be85-6598683c20ce" />
<img width="686" height="576" alt="cm_Stacking_val" src="https://github.com/user-attachments/assets/7c2aff7f-e27d-476b-ad60-83d81c2ed73e" />
<img width="686" height="576" alt="cm_Stacking_holdout" src="https://github.com/user-attachments/assets/46c06f12-c815-4128-8c84-c1c1bb14e381" />
<img width="668" height="576" alt="cm_RandomForest_val" src="https://github.com/user-attachments/assets/8629da0e-91fb-4e87-a30e-c1eee83c5e50" />
<img width="668" height="576" alt="cm_RandomForest_holdout" src="https://github.com/user-attachments/assets/1365e94e-915b-440e-a0b9-344a4d228a14" />
<img width="668" height="576" alt="cm_LogReg_val" src="https://github.com/user-attachments/assets/652fdf98-8ce9-460f-987c-72838624b2cd" />
<img width="668" height="576" alt="cm_LogReg_holdout" src="https://github.com/user-attachments/assets/eb5e28de-3cec-4e20-b036-a0c0312be406" />
<img width="668" height="576" alt="cm_GBM_val" src="https://github.com/user-attachments/assets/f8da681d-ad1a-4ec0-9ea8-249552f6f4b2" />
<img width="668" height="576" alt="cm_GBM_holdout" src="https://github.com/user-attachments/assets/f9bc1f5a-9845-4a0e-b5a9-9db2dfcc61d9" />
<img width="668" height="576" alt="cm_DNN_val" src="https://github.com/user-attachments/assets/35cd6bf9-bf8e-43e0-af09-05f2c066f9b5" />
<img width="668" height="576" alt="cm_DNN_holdout" src="https://github.com/user-attachments/assets/09f7dff6-89f9-472b-b473-414e1c2ea4fd" />
<img width="1058" height="817" alt="calibration_val" src="https://github.com/user-attachments/assets/97b16fc7-951d-4f35-bbf6-7785d2e2ae7c" />
<img width="1058" height="817" alt="calibration_test" src="https://github.com/user-attachments/assets/047a247f-a0b5-48aa-916d-0e97e8e9a2cf" />
<img width="1058" height="817" alt="calibration_holdout" src="https://github.com/user-attachments/assets/eb6f03d7-8c5f-47aa-8d39-f8b8850211b4" />
</p>
