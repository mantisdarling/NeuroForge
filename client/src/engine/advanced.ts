/** Bounded learning studies for the Signal Observatory. All training executes locally through the custom tensor engine. */
import { MNIST_MINI, decodeMnistMini } from "./mnistMini";
import { Adam, Conv2D, Linear, SGD, Sequential, Tanh, createRandom, crossEntropy } from "./nn";
import { Tensor } from "./tensor";
import { TrainingSession, type DatasetId, type OptimizerName } from "./training";

const oneHot = (labels: number[], classes: number) => labels.flatMap((label) => Array.from({ length: classes }, (_value, index) => (index === label ? 1 : 0)));
const argmax = (values: number[]) => values.reduce((best, value, index) => value > values[best] ? index : best, 0);

export interface StudySnapshot { epoch: number; loss: number; accuracy: number; history: number[]; }

export class DigitLabSession {
  readonly samples = MNIST_MINI.map((sample) => ({ label: sample.label, pixels: decodeMnistMini(sample.pixels) }));
  readonly model: Sequential;
  readonly optimizer: Adam;
  epoch = 0;
  history: number[] = [];

  constructor(seed = 922) {
    const random = createRandom(seed);
    this.model = new Sequential([new Linear(196, 24, random, "digit.input"), new Tanh(), new Linear(24, 10, random, "digit.output")]);
    this.optimizer = new Adam(this.model.parameters(), 0.025);
  }

  private inputs(): Tensor { return Tensor.from(this.samples.flatMap((sample) => sample.pixels), [this.samples.length, 196]); }
  private targets(): Tensor { return Tensor.from(oneHot(this.samples.map((sample) => sample.label), 10), [this.samples.length, 10]); }
  private evaluate(): StudySnapshot {
    return Tensor.noGrad(() => {
      const logits = this.model.forward(this.inputs());
      const probabilities = logits.softmax().toArray();
      const accuracy = this.samples.filter((sample, index) => argmax(probabilities.slice(index * 10, index * 10 + 10)) === sample.label).length / this.samples.length;
      return { epoch: this.epoch, loss: crossEntropy(logits, this.targets()).item(), accuracy, history: [...this.history] };
    });
  }

  train(epochs: number): StudySnapshot {
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      this.optimizer.zeroGrad();
      const loss = crossEntropy(this.model.forward(this.inputs()), this.targets());
      loss.backward();
      this.optimizer.step();
      this.epoch += 1;
      this.history.push(this.evaluate().loss);
    }
    return this.evaluate();
  }

  predict(pixels: number[]): number[] {
    if (pixels.length !== 196) throw new Error("Digit prediction expects a 14×14 grayscale drawing.");
    return Tensor.noGrad(() => this.model.forward(Tensor.from(pixels, [1, 196])).softmax().toArray());
  }
}

export interface OptimizerTrace { name: OptimizerName; history: number[]; finalLoss: number; }

export function compareOptimizers(dataset: DatasetId = "xor", width = 8, epochs = 48): OptimizerTrace[] {
  return (["sgd", "momentum", "adam"] as OptimizerName[]).map((name) => {
    const session = new TrainingSession(dataset, width, 320, { optimizer: name, batchSize: dataset === "xor" ? 4 : 16 });
    session.train(epochs);
    return { name, history: session.history, finalLoss: session.snapshot().loss };
  });
}

type ShapeLabel = 0 | 1 | 2;
const shapeNames = ["circle", "square", "triangle"] as const;

function makeShape(label: ShapeLabel, shiftRow: number, shiftColumn: number): number[] {
  return Array.from({ length: 36 }, (_value, index) => {
    const row = Math.min(5, Math.max(0, Math.floor(index / 6) - shiftRow));
    const column = Math.min(5, Math.max(0, index % 6 - shiftColumn));
    if (label === 0) return ((row - 2.5) ** 2 + (column - 2.5) ** 2 > 2.5 && (row - 2.5) ** 2 + (column - 2.5) ** 2 < 7.2) ? 1 : 0;
    if (label === 1) return row >= 1 && row <= 4 && column >= 1 && column <= 4 ? 1 : 0;
    return row >= 1 && row <= 4 && column >= 3 - (row - 1) && column <= 3 + (row - 1) ? 1 : 0;
  });
}

export class ShapeCnnSession {
  readonly labels: ShapeLabel[] = [];
  readonly inputs: number[][] = [];
  readonly conv: Conv2D;
  readonly classifier: Linear;
  readonly optimizer: Adam;
  epoch = 0;
  history: number[] = [];

  constructor(seed = 714) {
    for (let label = 0 as ShapeLabel; label < 3; label = (label + 1) as ShapeLabel) for (const [row, column] of [[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]]) { this.labels.push(label); this.inputs.push(makeShape(label, row, column)); }
    const random = createRandom(seed);
    this.conv = new Conv2D(1, 3, 3, random, "shapes.conv");
    this.classifier = new Linear(12, 3, random, "shapes.output");
    this.optimizer = new Adam([...this.conv.parameters(), ...this.classifier.parameters()], 0.028);
  }

  private inputTensor(): Tensor { return Tensor.from(this.inputs.flat(), [this.inputs.length, 1, 6, 6]); }
  private targetTensor(): Tensor { return Tensor.from(oneHot(this.labels, 3), [this.labels.length, 3]); }
  private logits(): Tensor { return this.classifier.forward(this.conv.forward(this.inputTensor()).relu().maxPool2d(2).flatten(1)); }

  train(epochs: number): StudySnapshot {
    for (let epoch = 0; epoch < epochs; epoch += 1) {
      this.optimizer.zeroGrad();
      const loss = crossEntropy(this.logits(), this.targetTensor());
      loss.backward();
      this.optimizer.step();
      this.epoch += 1;
      this.history.push(Tensor.noGrad(() => crossEntropy(this.logits(), this.targetTensor()).item()));
    }
    return this.snapshot();
  }

  snapshot(): StudySnapshot {
    return Tensor.noGrad(() => {
      const logits = this.logits();
      const probabilities = logits.softmax().toArray();
      const accuracy = this.labels.filter((label, index) => argmax(probabilities.slice(index * 3, index * 3 + 3)) === label).length / this.labels.length;
      return { epoch: this.epoch, loss: crossEntropy(logits, this.targetTensor()).item(), accuracy, history: [...this.history] };
    });
  }

  static names(): readonly string[] { return shapeNames; }
}
