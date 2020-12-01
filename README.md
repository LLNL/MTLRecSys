# Multitask Learning For Cancer
A project with the purpose of adapting multitask and state of the art ML recommender algorithms to predict cancer drug response.

## Examples and Blogs
* Example notebooks are in pipeline/walkthroughs
* These were used to create figures
* to see an example experiment look at pipeline/experiments/
    * MTL example
    * STL example

## Getting Started
1. Navigate to root directory, where environment.yml is located
2. conda env create -f environment.yml
3. conda activate mtl4c_env
4. Verify that environment installed correctly with conda env list

## Testing
* to run tests, go into pipeline directory and run pytest --cov=methods test_methods.py

## Documentation
* see mtl4cdocumentation.pdf 

## Experiments 
* stored in pipeline/experiments/
* see the readme in there for experiment descriptions

# Authors
- Alexander Ladd (ladd12@llnl.gov)
- André R. Gonçalves (goncalves1@llnl.gov)
- Braden C. Soper (soper3@llnl.gov)
- David P. Widemann (widemann1@llnl.gov)
- Pryiadip Ray (ray34@llnl.gov)

# CP Number: CP02373

# Dependencies and Licensing 
1. gpytorch (MIT)
2. pytorch (BSD)
3. keras (MIT)
4. Tensorflow (Apache License 2.0)
5. Surprise (BSD-3-Clause License)
6. SciKit Learn (New BSD License)

    
