---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======

M.Sc. in Machine Learning, University College London, 2024
------

* **Grade**: 84.88% &ndash; First Class (Honours)
* **Thesis Title** &ndash; On the Effects of DropEdge on Over-squashing in Deep GNNs
  
B.Sc. in Mathematical and Computer Sciences, Nanyang Technological University, 2023
------

* **Grade**: 4.58/5.00 &ndash; First Class (Honours)
* **Thesis Title** &ndash; Training-Free Neural Active Learning with Initialization-Robustness Guarantees
* **Notable Honours and Awards** &ndash;
  * 1st Prize in Integration Bee &ndash; NTU, Singapore, 2023
  * 3rd Prize in IET-Cup Hackathon &ndash; NTU, Singapore, 2022
  * 1st Prize in Integration Bee &ndash; NTU, Singapore, 2022
  * 1st Prize in Electronic Trading Challenge &ndash; Jane Street Capital, 2021
  * 3rd Prize in International Mathematics Competition &ndash; UCL, England, 2021
  * President Research Scholar &ndash; NTU, Singapore, 2021

AISSC in Science Stream, Venkateshwar International School, India, 2019
------

* **Grade** &ndash; 96.4%
* **Notable Honours and Awards** &ndash;
  * JEE Advance Scholarship &ndash; FIITJEE, India, 2019 - 2023
  * KVPY Scholarship &ndash; DST, Government of India, 2019
  * KVPY Scholarship &ndash; DST, Government of India, 2018
  * NTS Scholarship &ndash; NCERT, Government of India, 2017
  * JSTS Scholarship &ndash; DoE, Government of NCT of Delhi, 2016

Publications
======

<ul>{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}</ul>

Research Experience
======

Learning and Signal Processing Lab, UCL, Mar 2023 &ndash; Oct 2024
------

<ul>
  <li>Theoretically characterized the negative impact of DropEdge, DropNode, DropAgg, and DropGNN on over-squashing, suggesting their unsuitability for long-range tasks</li>
  <li>Empirically demonstrated the detrimental effects of random edge-dropping on test-time performance with heterophilic datasets: Squirrel, Chameleon and TwitchDE</li>
</ul>

Visual Cognitive Neuroscience Lab, NTU, Singapore, May 2023 &ndash; Oct 2023
------

<ul>
  <li>Developed LingML, a novel approach integrating linguistics into machine learning for fake news detection, achieving 80.9% accuracy solely using linguistic features</li>
  <li>Conducted an experimental study with a COVID-19 fake news dataset, demonstrating improvement in 11 transformers-based LLMs upon incorporation of linguistic features</li>
</ul>

MapleCG Lab, NUS, Singapore, Jun 2022 &ndash; May 2023
------

<ul>
  <li>Introduced a training-free active learning (AL) criterion, Expected Variance with Gaussian Processes (EV-GP), for neural networks in the NTK regime</li>
  <li>Proposed several AL algorithms using the EV-GP criterion, and benchmarked them against BADGE, MLMOC and K-Means++ algorithms on (E)MNIST, SVHN, CIFAR-100 and various UCI datasets</li>
</ul>

Employment Experience
======

Product Science Intern at Indeed Inc., Singapore, May 2022 &ndash; Aug 2022
------

<ul>
  <li>Statistically analysed user behaviour in the two groups of an A/B test to recommend next steps</li>
  <li>Modelled the top 20 international markets to identify a minimal criterion for screening resumes</li>
</ul>

Research Engineering Intern at Shopee Pte. Ltd., Singapore Jan 2022 &ndash; May 2022
------

<ul>
  <li>Engineered features resulting in 11.79% increase in total orders and 12.48% in orders per user in Brazil</li>
  <li>Reproduced Lin et al., 2017, and Kendall et al., 2017, for multi-task learning with data imbalance, leading to a 0.3% gain in CTR and 2% in CR in Malaysia</li>
  <li>Discovered unexpected ranking behaviour and identified the responsible feature through an ablation study</li>
</ul>

Full Stack Data Science Intern at Navtech Pte. Ltd., Singapore Jul 2020 &ndash; Aug 2020
------

<ul>
  <li>Built a recommendation system to be deployed as a B2B service to jewellery retailers</li>
  <li>Employed Neural Collaborative Filtering (NCF) to learn user-user and item-item similarities</li>
  <li>Proposed a solution to the cold start problem by engaging with the user before recommending to them</li>
</ul>

Teaching Experience
======

<ul>{% for post in site.teaching reversed %}
  {% include archive-single-cv.html %}
{% endfor %}</ul>
  
Volunteering Experience
======

Reviewer &ndash; International Conference on Learning Representations (ICLR), 2024

Certifications
======

<ul>
  <li>Applied Social Network Analysis in Python &ndash; University of Michigan, 2021</li>
  <li>Deep Learning Specialization &ndash; DeepLearning.AI, 2021</li>
  <li>AI Engineering Specialization &ndash; IBM, 2020</li>
  <li>Algorithms: Design and Analysis &ndash; Stanford University, 2020</li>
</ul>

Skills
======

<ul>
  <li>**Programming Languages** &ndash; Python, SQL, R, MATLAB</li>
  <li>**Frameworks and Tools** &ndash; PyTorch, Tensorflow, Hadoop (HDFS, YARN), Spark, Airflow</li>
</ul>