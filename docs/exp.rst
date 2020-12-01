Python Script Example
***********************



Python Script Example
========================

.. code-block:: python

    import sys
    sys.path.append('../')
    from design import ModelTraining
    from methods.mtl.MF_MTL import MF_MTL
    from methods.matrix_factorization.MF_STL import MF_STL

    # from methods.regressor.FFNN import FeedForwardNN
    from methods.matrix_factorization.MF import SVD_MF, NonNegative_MF
    from methods.knn.KNN import KNN_Normalized
    from datasets.DrugCellLines import DrugCellLinesMTL


    if __name__ == '__main__':

        drug_transform = {'type': 'pca', 'num_comp': 10}
        cell_transform = {'type': 'pca', 'num_comp': 10}
        dataset = DrugCellLinesMTL(['CCLE', 'GDSC', 'CTRP', 'NCI60'], common=True,
                                   unseen_cells=False, normalize=True,
                                   test_split=0.2, drug_transform=drug_transform,
                                   cell_transform=cell_transform)
        dataset.prepare_data()



        methods = [SVD_MF(n_factors=100),
                   KNN_Normalized(k=10)]

        metrics = ['rmse', 'explained_variance_score', 'mae']

        exp_folder = __file__.strip('.py')
        exp = ModelTraining(exp_folder)
        exp.execute(dataset, methods, metrics, nruns=1)
        exp.generate_report()
        
        
        