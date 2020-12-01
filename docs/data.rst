Data
************


Synthetic Data
===================
We can't publish real data, so instead there is a file that will load synthetic data. This file and it's attributes are important for adapting this software. If you want to use this code, the best way would be to create a loader class with similar methods to the methods outlined below.


.. automodule:: datasets.SyntheticData
   :members:

   
Example
==========

.. code-block:: python
    
    from datasets import SyntheticData as SD
    
    dataset = SD.SyntheticDataCreator(num_tasks=3,cellsPerTask=400, drugsPerTask=10, function="cosine",
                 normalize=False, noise=1, graph=False, test_split=0.3)
                 
    dataset.prepare_data() 
    
Now that we have instantiated dataset object, we can use dataset.data as a dictionary to access all the data 

.. code-block:: python

        task0_train_x = dataset.data['train']['x']['0']
        task0_train_y = dataset.data['train']['y']['0']
        task0_test_x = dataset.data['test']['x']['0']
        task0_test_y = dataset.data['test']['y']['0']
        
        
        