# Sudoku-Solver

A deep learning approach to solving Sudoku puzzles.

## Findings

**1. Custom Loss Function Improves Convergence**

The use of a custom loss function that explicitly guides training according to Sudoku rules significantly accelerates the learning process.

**Comparison (100k samples, 30 Epochs, ~4 min training time):**

* **Cross-Entropy Loss Only:** 11% test set accuracy
* **Hybrid Loss:** 76% test set accuracy

## Experiments

This section details the various experiments conducted to optimize the Sudoku solver model. Each experiment explores different architectural changes, training strategies, and loss functions to improve performance.

**1. Basic Conv2D Solution**

* **Dataset:** 900k samples
* **Epochs:** 1
* **Test Set Accuracy:** 80%

**2. Architecture Update (Row, Column, and Box Awareness)**

* **Description:** Modify the architecture to explicitly learn Sudoku constraints.
* **Dataset:** 900k samples
* **Epochs:** 1
* **Test Set Accuracy:** 82%

**3. Deeper Model**

* **Description:** Investigate the impact of increasing model depth.
* **Dataset:** 900k samples
* **Epochs:** 1
* **Test Set Accuracy:** 82%

**4. Residual Connections**

* **Description:** Implement residual connections to aid in training deeper networks.
* **Dataset:** 900k samples
* **Epochs:** 1
* **Test Set Accuracy:** 83%

**5. Progressive Learning (Difficulty-Based Training)**

* **Description:** Train the model on easier puzzles first, gradually increasing difficulty.
* **Dataset:** 900k samples
* **Epochs:** 1
* **Test Set Accuracy:** 83%

**6. Increased Epochs**

* **Description:** Evaluate the effect of training for a longer duration.
* **Dataset:** 900k samples
* **Epochs:** 5
* **Test Set Accuracy:** 86%

**7. Further Increase in Epochs with Early Stopping**

* **Description:** Observe the impact of even longer training with early stopping to prevent overfitting.
* **Dataset:** 900k samples
* **Epochs:** 10 (EarlyStopping applied)
* **Test Set Accuracy:** 86% (Training often stopped early)

**8. Regularization**

* **Description:** Add regularization techniques to improve generalization.
* **Dataset:** 900k samples
* **Epochs:** 10
* **Test Set Accuracy:** 86%

**9. Increased Training Data**

* **Description:** Assess the benefit of a larger training dataset.
* **Dataset:** 1.8M samples
* **Epochs:** 10
* **Test Set Accuracy:** 88%

**10. Pre-training on Puzzle Solutions (Target Only)**

* **Description:** Explore pre-training the model solely on solved Sudoku grids.
* **Dataset:** 1.8M samples
* **Epochs:** 1
* **Test Set Accuracy:** 82%

**11. Shallower Model with More Data (15M vs 4M Parameters)**

* **Description:** Compare a shallower model trained on a significantly larger dataset.
* **Dataset:** 9M samples
* **Epochs:** 50
* **Test Set Accuracy:** 85%

**12. Even Shallower Model with More Data (4M vs 1M Parameters)**

* **Description:** Further investigate the trade-off between model depth and dataset size.
* **Dataset:** 9M samples
* **Epochs:** 50
* **Test Set Accuracy:** 85%

**13. Deeper Model with More Difficulty Bins**

* **Description:** Train a deeper model with a more granular categorization of puzzle difficulty.
* **Model Parameters:** 8M (vs 4M)
* **Difficulty Bins:** 10 (vs 3)
* **Dataset:** 5M samples
* **Epochs:** 30
* **Test Set Accuracy:** 90%

**14. Hybrid Loss Function (Cross-Entropy + Sudoku Rule Penalties)**

* **Description:** Incorporate Sudoku rule awareness directly into the loss function.
* **Dataset:** 500k samples
* **Epochs:** 30
* **Test Set Accuracy:** 80% (Faster convergence observed)

**15. FixedNumberLayer**

* **Description:** Experiment with a layer that forces predictions to match the fixed numbers in the puzzle.
* **Dataset:** 100k samples
* **Epochs:** 30
* **Test Set Accuracy:** 54%

**16. Sudoku Rule Penalty Weight Scheduler**

* **Description:** Gradually increase the weight of the Sudoku rule penalties during training.
* **Dataset:** 1.5M samples
* **Epochs:** 100
* **Test Set Accuracy:** 87%

**17. Fixed Number Penalty in Loss Function**

* **Description:** Penalize the model when its predictions don't align with the initially given numbers.
* **Dataset:** 1.5M samples
* **Epochs:** 100
* **Test Set Accuracy:** 90%

**18. Adjusted Hybrid Loss Function**

* **Description:** Fine-tune the weights of different components in the hybrid loss function:
    * Empty Cell Cross-Entropy (CE) - Weight 1
    * Fixed Cell MSE - Weight 10
    * Sudoku Rules MSE - Weight 0.1
* **Dataset:** 100k samples
* **Epochs:** 30
* **Test Set Accuracy:** 76% (Achieved after only 4 minutes of training)
* 
* **Dataset:** 9M samples
* **Epochs:** 10
* **Test Set Accuracy:** 90%

**19. Create mixed datasets**

* **Description:** To avoid catastrophic forgetting, each difficulty dataset (10) is mixed with difficulties that model has already seen (80% vs. 20%).
* **Dataset:** 200k samples
* **Epochs:** 30
* **Test Set Accuracy:** 75% (Achieved after only 8 minutes of training)
* 
* **Dataset:** 9M samples
* **Epochs:** 10
* **Test Set Accuracy:** 86%

**20. Train only on hardest puzzles**

* **Description:** Train only on most difficult puzzles (difficulty 10). Increase sudoku rules penalty weight and decrease fixed numbers penalty weight.
* **Result:** Easy puzzles have only ~20% accuracy. It seems Easy (Difficulty 1) and rest are too different.
* **Dataset:** 500k(50k) samples
* **Epochs:** 30
* **Test Set Accuracy:** 77% (Achieved after 15 minutes of training)

**21. Train only on all puzzles at once**

* **Description:** Do not use curriculum learning.
* **Result:** Generalization is good but fixed numbers are weak (might be because of lower weight). However, model is still improving so more training will help.
* **Dataset:** 50k samples
* **Epochs:** 30
* **Test Set Accuracy:** 72% (Achieved after 14 minutes of training)