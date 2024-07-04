# water_distributed_contaminant

Requirement 
============
Python 3.12.2

Install library EPyt
1. pip install epyt
2. dont forget install other library requirement every script

or you can use Colab running this *.py script

===================================================================================================================================
Evaluating the performance of algorithms for sensor placement in a contamination detection network involves several metrics and techniques. Here are some key metrics and methods:
Metrics for Evaluation

    Fitness (Objective Function Value):
        Description: Measures how well the sensor placement achieves the objective, typically minimizing detection time or maximizing detection accuracy.
        Calculation: Already implemented as fitness_function.

    Coverage:
        Description: The percentage of nodes that are detected by the sensors.
        Calculation: Already implemented as calculate_coverage.

    Execution Time:
        Description: The time taken by the algorithm to find a solution.
        Calculation: Measured using time.time() before and after the algorithm runs.

Additional Metrics

    False Positives and False Negatives:
        Description: The number of nodes incorrectly detected as contaminated (false positives) and the number of contaminated nodes missed (false negatives).
        Calculation: Compare the detected nodes to the actual contaminated nodes.

    Precision and Recall:
        Description: 
        Precision is the proportion of correctly detected contaminated nodes out of all detected nodes. 
        Recall is the proportion of correctly detected contaminated nodes out of all actual contaminated nodes.
        Calculation:
            Precision = True Positives / (True Positives + False Positives)
            Recall = True Positives / (True Positives + False Negatives)