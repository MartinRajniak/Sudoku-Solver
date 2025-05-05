# Sudoku-Solver
Deep learning used to solve Sudoku puzzles.

## Experiments

1. Basic Conv2D solution
900k samples - 1 Epoch - 80% test set accuracy

2. Update architecture to allow model learn specific Sudoku features (Row, Column and Box constraint awareness)
900k samples - 1 Epoch - 82% test set accuracy

3. Deeper model 
900k samples - 1 Epoch - 82% test set accuracy

4. Residual connections
900k samples - 1 Epoch - 83% test set accuracy

5. Progressive learning (first train on easy puzzles and gradually increase difficulty)
900k samples - 1 Epoch - 83% test set accuracy

6. Increase epochs to 5
900k samples - 5 Epochs - 86% test set accuracy

7. Increase epochs to 10
900k samples - 10 Epochs - 86% test set accuracy (EarlyStopping usually stopped training early)

8. Add regularization
900k samples - 10 Epochs - 86% test set accuracy

9. Use 20% of training set
1800k samples - 10 Epochs - 88% test set accuracy

10. Pre-training on puzzle solutions (target) only
1800k samples - 1 Epoch - 82% test set accuracy

11. Full train set but with shallower model (15M vs 4M params)
9000k samples - 50 Epochs - 85% test set accuracy 

12. Full train set but with even shallower model (4M vs 1M params)
9000k samples - 50 Epochs - 85% test set accuracy

13. Deeper model (4M vs 8M) using more difficulty bins (3 vs. 10)
5000k samples - 30 Epochs - 90% test set accuracy

14. Use hybrid loss function mixing Cross-Entropy and Sudoku Rule Penalties
500k samples - 30 Epochs - 80% test set accuracy (faster convergence)

15. Add FixedNumberLayer to replace predictions with fixed numbers
100k samples - 30 Epochs - 54% test set accuracy

16. Add Fixed Number penalty to loss function when predictions don't match fixed numbers
???

17. Add Sudoku Rule Penalty Weight Scheduler to gradually increase constraint weight in loss function
1500k samples - 100 Epochs - 87% test set accuracy