# The code from 2019

This is the project as it was in May 2019, kept without changes (Keras, TensorFlow 1, Python 3.6): it does not run on current libraries.

It contains the bug described in the main [README](../README.md): `cube_tic_tac_toe_3d/CubeTicTacToeGame.py` returns the result of the game without looking at `player`, while `framework/Coach.py` relies on it.

The framework in this folder is based on [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair (MIT, see `LICENSE-alpha-zero-general`). `pytorch_classification/` and `utils/` come from the same project.
