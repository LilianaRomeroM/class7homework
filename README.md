# class7homework
REFACTOR DATASET_REGRESSION
IMPORTANT: These notes/observations still correspond to an analysis in progress.  Functions and calculation details will be provided in a later document.
  
a) For a new patient diagnosed with diabetes, could we use this information to determine key variables to control in order to mitigate its progression?
According to the preliminary analysis of the sample, the information does have significance and could be used to predict/mitigate disease progression in newly diagnosed patients.

*Which variables have the greatest effect on the diabetes progression?
In technical terms, the idea is to analyze whether there is a significant incidence of any factor on progression, in order to focus treatment efforts to prevent major
complications.
According to the initial information analyzed (see LASSO, alpha=0.1), we have two variables with high influence on disease progression: BMI and S5, followed by BP and S3.

b) Is the diabetes progression significantly influenced by the age or gender of the 
patient? (derived)
According to the same diagram of the previous point, gender has greater significance vs. disease progression than age.  Its weight as a predictor is not the highest among the variables but it does present visible variations of behavior with respect to both the BMI and the progression of the disease.
