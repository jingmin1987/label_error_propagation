# Label Error Propagation

## Problem Statement
The purpose of this project is to properly simulate and study the endogeneous properties related to label error propagation process which is similar to the process in a semi-supervised learning. Essentially, I'd like to study if the label revealation process is imperfect (e.g., due to human errors, data corruption, etc.), how this imperfection could impact the model which is trained on this imperfectly labeled data. Additionally, if this imperfection is sourced from model itself (e.g., using model to label new data), then how the future iterations of the model can be impacted. 

The main motivation behind this study is from my work. And the value from this study could help me identify opportunities in building a better framework between models and operations.

## Mathematical Abstraction
Let's consider a data generation process where predictions of an estimator is involved gradually throughout the process. 
1. We start with a dataset $D(X,y)$ generated from an unknown distribution where $X$ and $y$ are features and **ground truth** labels respectively
2. An initial estimator (e.g. a classifier model) $\theta_{(X, y)}$ is then properly trained on this dataset. The bias or error rate of the model is noted as $\varepsilon_{(X, y)}$ which is non-zero
3. For a new batch of observations $X_{new}$ with ground truth $y_{\text{ground truth}}$ drawn from the same distribution, we apply the estimator $\theta_{(X, y)}$ with the optimal threshold to classify.
4. With the predicted label $\hat{y}_{new}$, we generate the new data $D(X_{new}, y_{new})$ following the below process  

$$\begin{equation}
  \begin{cases} 
  y_{\text{ground truth}} & \text{if }\hat{y}_{new}=0\\
  y_{\text{ground truth}} & \text{if }\hat{y}_{new}=1\text{ and }randu()<\alpha\\
  \hat{y}_{new} & otherwise
  \end{cases} 
\end{equation}$$

   Here, one can assume that $\alpha$ is the probability that an operation analyst is able to reveal the ground truth 

5. Update the estimator based on the new dataset $D(X_{new}, y_{new})$ to obtain estimator $\theta_{(X_{new}, y_{new})}$. The update process could be incremental, meaning the new estimator $\theta_{(X_{new}, y_{new})}$ is trained on the combination of $D(X, y)$ and $D(X_{new}, y_{new})$, or completely retrained on $D(X_{new}, y_{new})$
6. Evalute the new estimator $\theta_{(X_{new}, y_{new})}$ on $D(X_{new}, y_{\text{ground truth}})$
7. Repeat step 3 to 6 multiple times and evaluate the performance deterioation, and the deviation of $y_{new}$ and $y_{\text{ground truth}}$ in each iteration

## Hypterparameters of the Study
* Estimator's bias $\varepsilon$. This is mainly determinted by the difficulty of the dataset, the algorithm used and the setup of hyperparameter tuning
* Size of datasets $D(X, y)$ and $D(X_{new}, y_{new})$
* Probability of revealing the ground truth: $\alpha$
* Updating mechanism - incremental update or replace

