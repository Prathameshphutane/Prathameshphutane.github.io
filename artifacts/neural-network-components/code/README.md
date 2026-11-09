# Experiment code

These scripts reproduce every number and chart in the *Inside a Neural Network* presentation.

- `playground_port.py` is a Python port of the TensorFlow Playground training code (`dataset.ts` and `nn.ts` from [tensorflow/playground](https://github.com/tensorflow/playground)). It uses the same four datasets, input features, weight initialization, squared-error loss and mini-batch gradient descent as the Playground.
- `run_experiments.py` trains 20 network setups, 5 times each with different random seeds, and saves the median results to `results.json`.
- `make_figures.py` draws the decision-boundary and loss-curve charts into an `img/` folder.

## Run it

```bash
pip install numpy matplotlib
python run_experiments.py   # about one minute
python make_figures.py
```

Playground defaults used unless a setting is being tested: 500 points, 50% training data, batch size 10, learning rate 0.03, tanh activation and a [4, 2] hidden-layer network.
