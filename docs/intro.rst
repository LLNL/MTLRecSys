Introduction
******************


Background
============
The goal of this code is to accurately model the interactions between cancer drugs and cancer cell lines across multiple datasets. Datasets can be from:

* `National Cancer Institute (NCI) <https://www.cancer.gov/>`_ :cite:`NCI`
* `Broad Institute Cancer and Cell Line Encyclopedia <https://portals.broadinstitute.org/ccle>`_ :cite:`CCLE`
* `Genomics and Drug Sensitivity in Cancer <https://www.cancerrxgene.org/>`_ 
* `National Institute of Health Clincial Trials Reporting Program <https://www.cancer.gov/about-nci/organization/ccct/ctrp>`_:cite:`NCI`

Problem Statement
==================
We need to have each of the following things from each dataset: a sparse ratings/interaction matrix indicating the (IC50) effective concentrations of each drug for each type of cancer and features for each respective drug and cell line. Here are some examples of services that generate theses features:

* `Dragon7 (discontinued) <https://chm.kode-solutions.net/products_dragon.php>`_
* `MOE <https://www.chemcomp.com/Products.htm>`_
* `Lincs Cell Features <http://www.lincsproject.org/>`_

Then we try to fit machine learning models to this data **to create valid predictors what the outcome of future drug and cancer line interactions will be** in order to inform future experiments and clinical doctors. Finally, in this introduction it will be useful to outline what ML methods we are using and some basic properties.


Methods
==================


.. list-table:: ML Methods Used
   :widths: 50 25 25 25
   :header-rows: 1

   * - Method
     - MTL or STL
     - Feature Based?
     - Source
     
   * - Collaborative Filtering Matrix Factorization
     - STL and MTL
     - Feature Based
     - Surprise :cite:`Hug2020`
   * - K Nearest Neighbors
     - STL
     - Not Feature Based
     - Surprise :cite:`Hug2020`
   * - Nonnegative Matrix Factorization
     - STL
     - Feature Based
     - Surprise :cite:`Hug2020`
   * - Feedforward Neural Net
     - STL
     - Feature Based
     - Custom w/ Pytorch :cite:`NEURIPS2019_9015`
   * - Gaussian Process
     - STL and MTL
     - Feature Based
     - Gpytorch :cite:`gardner2018gpytorch`
   * - Neural Collaborative Filtering
     - STL and MTL
     - Both
     - Author Github :cite:`NCF`






