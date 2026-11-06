# Third-party notices

The code written for this project is under the [MIT license](LICENSE). Parts of the 2019 code in `legacy/` come from other open-source projects and stay under their own licenses.

## alpha-zero-general

The 2019 framework in `legacy/` (`framework/`, `othello/`, `tic_tac_toe/`, and the structure of the 3D games) is based on [alpha-zero-general](https://github.com/suragnair/alpha-zero-general) by Surag Nair and contributors, including Shantanu Kumar (TensorFlow and Keras models for Othello) and Evgeny Tyurin (rules and a model for tic-tac-toe). MIT license, see [`legacy/LICENSE-alpha-zero-general`](legacy/LICENSE-alpha-zero-general), kept as it read when the code was taken in 2019. The current notice of that project is dated 2025.

The code outside `legacy/` was written from scratch, but the idea of its training loop comes from the same place, and alpha-zero-general is where this project started.

## pytorch-classification

`legacy/pytorch_classification/` comes from [pytorch-classification](https://github.com/bearpaw/pytorch-classification) by Wei Yang. MIT license, see [`legacy/LICENSE-pytorch-classification`](legacy/LICENSE-pytorch-classification).

## progress

`legacy/utils/progress/` and `legacy/pytorch_classification/utils/progress/` come from [progress](https://github.com/verigak/progress) by Giorgos Verigakis. ISC license, see [`legacy/utils/progress/LICENSE`](legacy/utils/progress/LICENSE).

## Ideas

The method follows AlphaZero: Silver et al., "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play", Science, 2018.
