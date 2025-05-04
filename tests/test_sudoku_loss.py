# --- Run `python -m unittest tests.test_sudoku_loss` in root dir

import unittest

import tensorflow as tf

import numpy as np

from sudoku_solver.model.loss import SudokuLoss

from tests.test_data import *

TOTAL_PUZZLE_NUMBERS = 81

# How many decimal places points to compare
PRECISION = 10**4


def _convert_to_target_tensor(y):
    # Convert digits to one-hot encoded format (perfect predictions)
    target = np.zeros((1, 9, 9, 1))  # One batch
    for i in range(9):
        for j in range(9):
            digit = y[i, j] - 1  # Convert 1-9 to 0-8
            target[0, i, j, 0] = digit

    target_tensor = tf.constant(target, dtype=tf.float32)
    return target_tensor


def _convert_to_predictions_tensor(y):
    # Convert digits to one-hot encoded format (perfect predictions)
    predictions = np.zeros((1, 9, 9, 9))  # One batch
    for i in range(9):
        for j in range(9):
            digit = y[i, j] - 1  # Convert 1-9 to 0-8
            predictions[0, i, j, digit] = 1.0

    predictions_tensor = tf.constant(predictions, dtype=tf.float32)
    return predictions_tensor


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.sudoku_loss = SudokuLoss()

    def test_everything_ok(self):
        predictions_tensor = _convert_to_predictions_tensor(VALID_SOLUTION)

        row_penalty = self.sudoku_loss._row_penalty(predictions_tensor)
        column_penalty = self.sudoku_loss._column_penalty(predictions_tensor)
        box_penalty = self.sudoku_loss._box_penalty(predictions_tensor)

        self.assertEqual(int(row_penalty * PRECISION), 0)
        self.assertEqual(int(column_penalty * PRECISION), 0)
        self.assertEqual(int(box_penalty * PRECISION), 0)

    def test_one_row_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_ROW_TWO_NUMBERS)

        row_penalty = self.sudoku_loss._row_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(row_penalty * PRECISION), int(2 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_one_row_three_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_ROW_THREE_NUMBERS)

        row_penalty = self.sudoku_loss._row_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(row_penalty * PRECISION), int(6 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_two_rows_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_TWO_ROWS_TWO_NUMBERS)

        row_penalty = self.sudoku_loss._row_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(row_penalty * PRECISION), int(4 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_one_column_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(
            WRONG_ONE_COLUMN_TWO_NUMBERS
        )

        column_penalty = self.sudoku_loss._column_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(column_penalty * PRECISION), int(2 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_one_column_three_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(
            WRONG_ONE_COLUMN_THREE_NUMBERS
        )

        column_penalty = self.sudoku_loss._column_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(column_penalty * PRECISION), int(6 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_two_columns_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(
            WRONG_TWO_COLUMNS_TWO_NUMBERS
        )

        column_penalty = self.sudoku_loss._column_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(column_penalty * PRECISION), int(4 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_one_box_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_BOX_TWO_NUMBERS)

        box_penalty = self.sudoku_loss._box_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(box_penalty * PRECISION), int(2 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_one_box_three_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_BOX_THREE_NUMBERS)

        box_penalty = self.sudoku_loss._box_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(box_penalty * PRECISION), int(6 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_two_boxes_two_numbers_wrong(self):
        predictions_tensor = _convert_to_predictions_tensor(WRONG_TWO_BOXES_TWO_NUMBERS)

        box_penalty = self.sudoku_loss._box_penalty(predictions_tensor)

        # Magic number: sum(square(number_occurrences - 1))
        self.assertEqual(
            int(box_penalty * PRECISION), int(4 / TOTAL_PUZZLE_NUMBERS * PRECISION)
        )

    def test_error_in_empty(self):
        target_tensor = _convert_to_target_tensor(VALID_SOLUTION)
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_ROW_TWO_NUMBERS)
        cross_entropy = self.sudoku_loss._masked_cross_entropy(
            target_tensor,
            predictions_tensor,
            MASK_ONE_ROW_TWO_NUMBERS.reshape(1, 9, 9),
        )
        self.assertGreater(int(cross_entropy * PRECISION), 0)

    def test_error_in_fixed(self):
        target_tensor = _convert_to_target_tensor(VALID_SOLUTION)
        predictions_tensor = _convert_to_predictions_tensor(WRONG_ONE_ROW_TWO_NUMBERS)
        cross_entropy = self.sudoku_loss._masked_cross_entropy(
            target_tensor,
            predictions_tensor,
            np.zeros((1, 9, 9)),
        )
        self.assertEqual(int(cross_entropy * PRECISION), 0)


if __name__ == "__main__":
    unittest.main()
