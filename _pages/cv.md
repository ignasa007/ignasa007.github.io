---
layout: archive
title: "CURRICULUM VITAE"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

<div style="margin-top: 50px;"></div>

<u><span style="font-variant:small-caps;">Education</span></u>
======

### M.Sc. in Machine Learning, University College London (2023 &ndash; 2024)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><b>Grade</b>: 84.88% &ndash; First Class (Honours)</li>
  <li><b>Thesis Title</b>: On the Effects of DropEdge on Over-squashing in Deep GNNs</li>
</ul>
  
### B.Sc. in Mathematical and Computer Sciences, Nanyang Technological University (2019 &ndash; 2023)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><b>Grade</b>: 4.58/5.00 &ndash; First Class (Honours)</li>
  <li><b>Thesis Title</b>: Training-Free Neural Active Learning with Initialization-Robustness Guarantees</li>
  <li><b>Notable Honours and Awards</b>:
    <ul style="list-style-type: circle; padding-left: 5mm;">
      <li><b>1st Prize in Integration Bee</b> &ndash; NTU, Singapore (2023)</li>
      <li><b>3rd Prize in IET-Cup Hackathon</b> &ndash; NTU, Singapore (2022)</li>
      <li><b>1st Prize in Integration Bee</b> &ndash; NTU, Singapore (2022)</li>
      <li><b>1st Prize in Electronic Trading Challenge</b> &ndash; Jane Street Capital (2021)</li>
      <li><b>3rd Prize in International Mathematics Competition</b> &ndash; UCL, England (2021)</li>
      <li><b>President Research Scholar</b> &ndash; NTU, Singapore (2021)</li>
    </ul>
  </li>
</ul>

### AISSC in Science Stream, Venkateshwar International School, India (2017 &ndash; 2019)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><b>Grade</b>: 96.4%</li>
  <li><b>Notable Honours and Awards</b>:
    <ul style="list-style-type: circle; padding-left: 5mm;">
      <li><b>JEE Advance Scholarship</b> &ndash; FIITJEE, India (2019 - 2023)</li>
      <li><b>KVPY Scholarship</b> &ndash; DST, Government of India (2019)</li>
      <li><b>KVPY Scholarship</b> &ndash; DST, Government of India (2018)</li>
      <li><b>NTS Scholarship</b> &ndash; NCERT, Government of India (2017)</li>
      <li><b>JSTS Scholarship</b> &ndash; DoE, Government of NCT of Delhi (2016)</li>
    </ul>
  </li>
</ul>

<u><span style="font-variant:small-caps;">Publications</span></u>
======

<ul style="list-style-type: disc; padding-left: 5mm;">
{% for post in site.publications reversed %}
  {% include archive-single-cv.html %}
{% endfor %}
</ul>

<u><span style="font-variant:small-caps;">Research Experience</span></u>
======

### Learning and Signal Processing Lab, UCL (Mar 2023 &ndash; Oct 2024)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Theoretically characterized the negative impact of DropEdge, DropNode, DropAgg, and DropGNN on over-squashing, suggesting their unsuitability for long-range tasks</li>
  <li>Empirically demonstrated the detrimental effects of random edge-dropping on test-time performance with heterophilic datasets: Squirrel, Chameleon and TwitchDE</li>
</ul>

### Visual Cognitive Neuroscience Lab, NTU, Singapore (May 2023 &ndash; Oct 2023)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Developed LingML, a novel approach integrating linguistics into machine learning for fake news detection, achieving 80.9% accuracy solely using linguistic features</li>
  <li>Conducted an experimental study with a COVID-19 fake news dataset, demonstrating improvement in 11 transformers-based LLMs upon incorporation of linguistic features</li>
</ul>

### MapleCG Lab, NUS, Singapore (Jun 2022 &ndash; May 2023)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Introduced a training-free active learning (AL) criterion, Expected Variance with Gaussian Processes (EV-GP), for neural networks in the NTK regime</li>
  <li>Proposed several AL algorithms using the EV-GP criterion, and benchmarked them against BADGE, MLMOC and K-Means++ algorithms on (E)MNIST, SVHN, CIFAR-100 and various UCI datasets</li>
</ul>

<u><span style="font-variant:small-caps;">Employment Experience</span></u>
======

### Product Science Intern at Indeed Inc., Singapore (May 2022 &ndash; Aug 2022)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Statistically analysed user behaviour in the two groups of an A/B test to recommend next steps</li>
  <li>Modelled the top 20 international markets to identify a minimal criterion for screening resumes</li>
</ul>

### Research Engineering Intern at Shopee Pte. Ltd., Singapore (Jan 2022 &ndash; May 2022)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Engineered features resulting in 11.79% increase in total orders and 12.48% in orders per user in Brazil</li>
  <li>Reproduced Lin et al., 2017, and Kendall et al., 2017, for multi-task learning with data imbalance, leading to a 0.3% gain in CTR and 2% in CR in Malaysia</li>
  <li>Discovered unexpected ranking behaviour and identified the responsible feature through an ablation study</li>
</ul>

### Full Stack Data Science Intern at Navtech Pte. Ltd., Singapore (Jul 2020 &ndash; Aug 2020)

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li>Built a recommendation system to be deployed as a B2B service to jewellery retailers</li>
  <li>Employed Neural Collaborative Filtering (NCF) to learn user-user and item-item similarities</li>
  <li>Proposed a solution to the cold start problem by engaging with the user before recommending to them</li>
</ul>

<u><span style="font-variant:small-caps;">Teaching Experience</span></u>
======

<ul style="list-style-type: disc; padding-left: 5mm;">
{% for post in site.teaching reversed %}
  {% include archive-single-cv.html %}
{% endfor %}
</ul>
  
<u><span style="font-variant:small-caps;">Volunteering Experience</span></u>
======

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><b>Reviewer</b> &ndash; International Conference on Learning Representations (2024)</li>
</ul>

<u><span style="font-variant:small-caps;">Certifications</span></u>
======

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><a href="https://www.coursera.org/account/accomplishments/certificate/LSJR2KMNRRUR"><strong>Applied Social Network Analysis in Python</strong></a> &ndash; University of Michigan (2021)</li>
  <li><a href="https://coursera.org/share/992110f4684c2aa3b9826f73804d4b1c"><strong>Deep Learning Specialization</strong></a> &ndash; DeepLearning.AI (2021)</li>
  <li><a href="https://www.youracclaim.com/badges/e79bf049-f3d8-45e6-8890-6b83b27b5d7a/linked_in_profile"><strong>AI Engineering Specialization</strong></a> &ndash; IBM (2020)</li>
  <li><a href="https://verify.lagunita.stanford.edu/SOA/0e460be2891a48d3a44d48b92d9531a8"><strong>Algorithms: Design and Analysis</strong></a> &ndash; Stanford University (2020)</li>
</ul>

<u><span style="font-variant:small-caps;">Skills</span></u>
======

<ul style="list-style-type: disc; padding-left: 5mm;">
  <li><b>Programming Languages</b> &ndash; Python, SQL, R, MATLAB</li>
  <li><b>Frameworks and Tools</b> &ndash; PyTorch, Tensorflow, Hadoop (HDFS, YARN), Spark, Airflow</li>
</ul>