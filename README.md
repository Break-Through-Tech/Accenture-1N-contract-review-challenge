# Accenture Contract Review Challenge

> 💡 **Note for the team:** This README will be updated throughout AI Studio as the project progresses. Sections that are still in development retain the original template guidance for future completion.

---

### 👥 **Team Members**

| Name           | GitHub Handle    | Contribution |
| -------------- | ---------------- | ------------ |
| Albert Yorn    | @albrtyrn        |              |
| Josiah De Leon | @josweon         |              |
| Alec Borque    | @AlecDerinBorque |              |

---

## 🎯 **Project Highlights**

**Example:**

* Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
* Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
* Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
* Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

This project is part of the **Break Through Tech AI Program** and is being completed in partnership with **Accenture**.

The goal of the project is to build a machine learning system that can assist with **contract review and risk identification**. Using the **Contract Understanding Atticus Dataset (CUAD)**, the project focuses on identifying important contractual clauses and supporting the automated analysis of legal agreements.

The project will explore a pipeline that can:

* Detect and classify clauses within contracts.
* Identify contract language associated with predefined legal categories.
* Evaluate model performance across different clause types.
* Support downstream contract triage and risk-review workflows.

Contract review is traditionally a time-intensive process requiring manual analysis of large legal documents. A reliable automated system could help reduce review time, surface potentially important clauses earlier, and support legal and business teams in prioritizing contracts that require closer examination.

---

## 📊 **Data Exploration**

The project uses the **Contract Understanding Atticus Dataset (CUAD)**, a legal contract dataset designed for machine learning research in contract review.

CUAD contains:

* **510 commercial contracts**
* **41 legal clause categories**
* Human-annotated contract clauses and associated labels
* Official training and testing splits

Initial work will focus on understanding the structure of the dataset, examining the distribution of clause categories, and identifying challenges such as **class imbalance** and variation in contract language.

**You might consider describing the following (as applicable):**

* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

The initial modeling phase will establish a simple baseline before exploring more advanced approaches.

The current planned baseline is:

* **TF-IDF features**
* **Logistic Regression classifier**
* Evaluation using clause-level predictions and classification metrics

Future modeling work may explore transformer-based NLP models or other approaches for improved contract clause classification.

**You might consider describing the following (as applicable):**

* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

Because CUAD contains significant class imbalance across clause categories, evaluation will prioritize metrics such as **precision, recall, and F1 score**, including performance at the individual clause-category level.

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

Current priorities include:

* Explore and understand the CUAD dataset structure.
* Build the initial TF-IDF + Logistic Regression baseline.
* Establish baseline precision, recall, and F1 metrics.
* Analyze performance across individual clause categories.
* Use baseline results to guide future model development.

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
