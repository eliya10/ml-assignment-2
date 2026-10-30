# PlateScope - Steel Plate Fault Classification

M.Tech (AIML) | Machine Learning | Assignment 2

## a. Problem statement

Classify a steel plate surface defect into one of seven fault categories using geometric, luminosity and steel-property measurements. Compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine on the same held-out test data. Provide an interactive Streamlit app that evaluates saved models on a labelled CSV without training on the upload.

## b. Dataset description

- Dataset: Steel Plates Faults, UCI Machine Learning Repository, dataset ID 198.
- Source: [UCI Steel Plates Faults](https://archive.ics.uci.edu/dataset/198/steel+plates+faults).
- Citation: Buscema, M., Terzi, S., and Tastle, W. (2010). Steel Plates Faults. UCI Machine Learning Repository. DOI: [10.24432/C5J88N](https://doi.org/10.24432/C5J88N).
- License: Creative Commons Attribution 4.0 International (CC BY 4.0). Dataset is redistributed with attribution; the seven one-hot target columns were converted to one named target column.
- Instances: 1,941. Input features: 27 numeric features. Both exceed the required minimums of 500 instances and 12 features. The target is not counted as an input feature.
- Target: `fault_type`, with seven classes. `K_Scatch` intentionally retains the UCI spelling.
- Missing values: zero. Duplicate rows and duplicate feature vectors: zero. No records were removed and no synthetic observations were added.
- Feature groups: bounding-box position; pixel area and perimeters; luminosity; conveyor length; steel type and thickness; edge, shape, orientation and area indices. All 27 original input features are retained.

| Fault class | Full dataset | Train | Test |
| --- | --- | --- | --- |
| Bumps | 402 | 321 | 81 |
| Dirtiness | 55 | 44 | 11 |
| K_Scatch | 391 | 313 | 78 |
| Other_Faults | 673 | 538 | 135 |
| Pastry | 158 | 126 | 32 |
| Stains | 72 | 58 | 14 |
| Z_Scratch | 190 | 152 | 38 |

The original `Faults.NNA` and `Faults27x7_var` files were obtained from the UCI download. `data/steel_plates_faults.csv` contains the converted dataset, so neither training nor the app needs an online data download.

**Experimental design.** A single stratified 80/20 split with fixed random seed 5801 produces 1,552 training rows and 389 held-out test rows. Every classifier uses precisely the same rows. Splitting precedes preprocessing. StandardScaler is fitted only on the training rows for Logistic Regression, KNN, Gaussian Naive Bayes and SVM; Decision Tree uses unscaled inputs. Each complete preprocessing/classifier pipeline is saved in `model/`. No oversampling or feature selection is applied. Hyperparameters are fixed baseline settings; the test set is not used for tuning. SHA-256 hashes and original source row indices are retained in `model/metadata.json`.

**Limitations.** Classes are imbalanced: Other_Faults has 673 rows and Dirtiness has 55. Macro metrics give every fault class equal weight. Results describe one random split of this historical dataset; they do not establish performance across production lines, future data or different splits. Future tuning should use cross-validation within the training partition and a fresh untouched final test set.

## c. Github Repository Link

Repository link: Not published yet.

The actual repository URL must be entered here after publishing, and the updated README must be committed and pushed before the final PDF is produced.

## d. Models used

| ML Model Name | Configuration | 
| --- | --- |
| Logistic Regression | StandardScaler; C=1.0; max_iter=3000; multinomial classification with the default solver. |
| Decision Tree | max_depth=8; min_samples_leaf=4; random_state=5801. |
| KNN | StandardScaler; n_neighbors=9; distance-weighted voting; Euclidean distance. |
| Naive Bayes | StandardScaler; GaussianNB; var_smoothing=1e-9. |
| Support Vector Machine | StandardScaler; RBF kernel; C=5.0; gamma=scale; probability=True; random_state=5801. |

**Comparison on the 389-row held-out test set.** Values below are rounded to four decimal places. Unrounded calculations are saved in `results/metrics.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7018 | 0.9218 | 0.7068 | 0.6967 | 0.7003 | 0.6150 |
| Decision Tree | 0.7249 | 0.8953 | 0.7529 | 0.7169 | 0.7301 | 0.6488 |
| KNN | 0.7224 | 0.9378 | 0.7356 | 0.7414 | 0.7371 | 0.6438 |
| Naive Bayes | 0.5835 | 0.9203 | 0.6044 | 0.7041 | 0.6121 | 0.5169 |
| Support Vector Machine | 0.7558 | 0.9552 | 0.7918 | 0.7449 | 0.7641 | 0.6839 |

**Metric definitions.** Accuracy is the fraction of correct predictions. Precision, recall and F1 are unweighted macro averages over the seven fault classes. AUC is the macro one-vs-rest ROC AUC using `predict_proba`, with probability columns aligned to the class labels. MCC is the multiclass Matthews correlation coefficient. Accuracy, precision, recall, F1 and AUC range from 0 to 1; MCC ranges from -1 to 1. Zero-division values in per-class precision/recall/F1 are set to zero. If an uploaded test CSV omits a class, the app displays AUC as N/A; macro precision/recall/F1 still average over all seven classes, with undefined per-class values set to zero.

### Observations about model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.7018 and macro F1 0.7003 provide a linear baseline. AUC 0.9218 indicates useful class ranking, but the lower F1 suggests that linear decision boundaries do not fully separate these fault categories. |
| Decision Tree | Accuracy 0.7249 and macro F1 0.7301 improve on the linear baseline. Its macro precision of 0.7529 exceeds KNN's 0.7356, while its AUC of 0.8953 is the lowest of the five. Leaf-based probability estimates and hard predictions describe different aspects of performance. |
| KNN | Macro F1 0.7371 and recall 0.7414 exceed the tree's 0.7301 and 0.7169, although accuracy 0.7224 is slightly below the tree's 0.7249. Scaling and local distance-weighted voting are useful for this feature space. |
| Naive Bayes | Lowest accuracy (0.5835) and macro F1 (0.6121). Other_Faults recall is only 0.2519, while Stains recall is 1.0000. AUC 0.9203 remains relatively high, illustrating that good probability ranking need not yield the best predicted class. Correlated measurements and non-Gaussian feature distributions are plausible limitations of this baseline, rather than proven causal findings. |
| Support Vector Machine | Best accuracy (0.7558), AUC (0.9552), macro precision (0.7918), macro recall (0.7449), macro F1 (0.7641) and MCC (0.6839) on this split. RBF boundaries capture useful nonlinear patterns. Bumps and Other_Faults remain confused: 20 Bumps rows become Other_Faults and 25 Other_Faults rows become Bumps. |
| Overall, Winner for your dataset? | Support Vector Machine, selected for the highest held-out macro F1 (0.7641); it also leads the other five reported metrics. This is a comparison on this split, not a cross-validated claim. |

### Project files

```text
app.py                         Streamlit evaluation frontend
common.py                      Schema, CSV validation and shared metrics
train.py                       Five models, split, preprocessing and outputs
verify_project.py              Saved-artifact and data-isolation checks
make_submission.py             Final single-PDF builder
requirements.txt               Pinned tested package versions
README.md                      This project explanation and comparison
.streamlit/config.toml         Theme and upload limit
test_data.csv                  389 labelled held-out test rows
model/                         Five fitted .joblib pipelines and metadata.json
data/steel_plates_faults.csv    Full attributed UCI dataset
data/train_data.csv            Training partition (never used for app evaluation)
results/                       Metrics, predictions, reports, chart and run output
```

### Run and verify in BITS Virtual Lab

This is an AI-assisted learning reference. The assignment permits AI for learning support, not direct copy-paste submissions. Review, understand and adapt the implementation, verify the observations, and retain an honest record of your own work. The bundled run outputs were produced during project preparation and are not BITS Virtual Lab execution evidence.

Use Python 3.12. In the project folder, run:

```bash
python -m pip install -r requirements.txt
python train.py
python verify_project.py
python -m streamlit run app.py
```

After training-code changes, rerun training and verification and update the README's comparison and observations to match the new outputs. Do not upload `data/train_data.csv` or the full source data for app evaluation. Download the bundled `test_data.csv` from the sidebar, or upload a CSV with exactly the 27 required numeric input columns and the `fault_type` target. Select a classifier to see six metrics, the five-model comparison, a confusion matrix, a per-class classification report and downloadable predictions. Uploaded values must be finite and non-missing. Unknown labels and unexpected columns are rejected. Uploads are limited to 5 MB and 5,000 test rows.

Take exactly one genuine screenshot showing execution within the BITS Virtual Lab. Keep the lab interface visible alongside the successful training/verification output or running application. A local screenshot cannot substitute for this requirement.

### Publish and deploy

Create a GitHub repository and commit the project files, including all five `.joblib` files, metadata, CSVs, requirements and README. Keep real commits reflecting your work; do not manufacture commit history. Update section c with the actual repository link and push it.

Open [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub, choose Create app, select the repository and branch, and set the entrypoint to `app.py`. In Advanced settings select Python 3.12. Deploy, then open the resulting `.streamlit.app` URL and check CSV upload, all five dropdown options, six metrics and the confusion matrix. Open both repository and app links in a separate browser session to verify evaluator access. See [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

The app loads repository-owned saved pipelines and never retrains on an uploaded CSV. There is no training control in the app. Never load untrusted model files; the upload control accepts CSV only.

### Create the required single submission PDF

After the repository and live app work and you have the genuine lab screenshot, run the builder with those actual values:

```bash
python make_submission.py --repository-url REAL_GITHUB_REPOSITORY_URL \
  --app-url REAL_STREAMLIT_APP_URL --lab-screenshot PATH_TO_REAL_LAB_SCREENSHOT
```

The builder requires valid provider URL formats, a real image file and the repository link in README; it runs artifact verification, then writes `submission/ML_Assignment_2.pdf`. It does not verify account access or certify the screenshot's origin, so check these yourself. The PDF preserves the required order: (1) clickable GitHub repository link, (2) clickable live Streamlit app link, (3) one BITS Virtual Lab screenshot, (4) the complete README content. The screenshot is included without alteration or cropping.

Submit only this completed PDF through the LMS. The project ZIP is a working package, not the LMS submission format. Review the final PDF, verify that both links work for the evaluator, and click SUBMIT before 8 November 2026, 23:59. Draft submissions are not accepted.
