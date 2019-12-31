from nasbench.lib import config
from nasbench.lib.evaluate import train_and_evaluate
from nasbench.lib.model_spec import ModelSpec

from absl import app


def main(argv):
    model_spec = ModelSpec([[0, 1, 1, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0]],
                           ['input', 'maxpool3x3', 'maxpool3x3', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'output']
                           )
    cfg = config.build_config()
    result = train_and_evaluate(model_spec, cfg, 'outputs')
    results = result['evaluation_results']
    for result in results:
        result.pop('sample_metrics')
        print(result)


if __name__ == "__main__":
    app.run(main)
