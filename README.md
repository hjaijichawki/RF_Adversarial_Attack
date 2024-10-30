*Adversarial Attacks Against Random Forest Classifier*
--------------------------------------

**Introduction**
------------------------

In recent years, there have been significant advancements in various industries due to the development of artificial intelligence. AI-enabled systems, ranging from self-driving cars to medical tests, have demonstrated impressive abilities. Nevertheless, with the increasing incorporation of AI in business tools and processes, adversarial attacks have quickly emerged as a new threat against these models. We will try here to implement an adversarial attacks against Random Forest based on the Decision Tree attack introduced by Papernot et al. in 2016.


**Dataset: Home Equity Line of Credit (HELOC)**
----------------

This competition focuses on an anonymized dataset of Home Equity Line of Credit (HELOC) applications made by real homeowners. A HELOC is a line of credit typically offered by a bank as a percentage of home equity (the difference between the current market value of a home and its purchase price). The customers in this dataset have requested a credit line in the range of $5,000 - $150,000. The fundamental task is to use the information about the applicant in their credit report to predict whether they will repay their HELOC account within 2 years. This prediction is then used to decide whether the homeowner qualifies for a line of credit and, if so, how much credit should be extended. 


**Papernot Attack**
-----------------------------

Papernot et al. introduce a novel adversarial attack targeting decision trees by exploiting the structure of decision trees. The attack creates adversarial samples by finding a misclassifying path from the original class leaf to a leaf with a different class. Specifically, the algorithm identifies conditions along the path to the adversarial leaf and modifies the input features so that it satisfies those conditions, thereby forcing the decision tree to classify the input incorrectly.

[https://arxiv.org/abs/1605.07277](https://arxiv.org/abs/1605.07277 )
