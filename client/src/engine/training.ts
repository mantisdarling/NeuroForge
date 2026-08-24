import { Adam, Linear, SGD, Sequential, Tanh, createRandom, crossEntropy, meanSquaredError } from "./nn";
import { Tensor } from "./tensor";

export type DatasetId = "xor" | "spiral" | "sine";
export type TrainingTask = "classification" | "regression";

export interface Dataset {
  id: DatasetId;
  name: string;
  task: TrainingTask;
  inputs: number[][];
  labels: number[];
  targets: number[][];
  inputFeatures: number;
  outputFeatures: number;
}

function oneHot(labels: number[], classes: number): number[][] {
  return labels.map((label) => Array.from({ length: classes }, (_, index) => (index === label ? 1 : 0)));
}

function makeSpiral(): Dataset {
  const random = createRandom(78);
  const points: number[][] = [];
  const labels: number[] = [];
  const samplesPerClass = 48;
  for (let label = 0; label < 2; label += 1) {
    for (let sample = 0; sample < samplesPerClass; sample += 1) {
      const radius = sample / samplesPerClass;
      const angle = label * Math.PI + radius * 3.4 + (random() - 0.5) * 0.26;
      points.push([radius * Math.sin(angle), radius * Math.cos(angle)]);
      labels.push(label);
    }
  }
  return { id: "spiral", name: "Two-class spiral", task: "classification", inputs: points, labels, targets: oneHot(labels, 2), inputFeatures: 2, outputFeatures: 2 };
}

const xorLabels = [0, 1, 1, 0];
const datasets: Record<DatasetId, Dataset> = {
  xor: {
    id: "xor",
    name: "XOR logic gate",
    task: "classification",
    inputs: [[0, 0], [0, 1], [1, 0], [1, 1]],
    labels: xorLabels,
    targets: oneHot(xorLabels, 2),
    inputFeatures: 2,
    outputFeatures: 2,
  },
  spiral: makeSpiral(),
  sine: (() => {
    const inputs = Array.from({ length: 40 }, (_, index) => -1 + (2 * index) / 39).map((value) => [value]);
    const targets = inputs.map(([value]) => [Math.sin(value * Math.PI)]);
    return { id: "sine", name: "Sine regression", task: "regression", inputs, labels: [], targets, inputFeatures: 1, outputFeatures: 1 };
  })(),
};

export function getDataset(id: DatasetId): Dataset {
  return datasets[id];
}

function flatten(rows: number[][]): number[] {
  return rows.flat();
}

export interface TrainingSnapshot {
  epoch: number;
  loss: number;
  accuracy?: number;
  output: number[];
  gradientMagnitude: number;
}

export type OptimizerName = "sgd" | "momentum" | "adam";
type StepOptimizer = { zeroGrad(): void; step(): void };

export interface TrainingOptions {
  optimizer?: OptimizerName;
  batchSize?: number;
}

export class TrainingSession {
  readonly dataset: Dataset;
  readonly model: Sequential;
  readonly optimizer: StepOptimizer;
  epoch = 0;
  history: number[] = [];
  batchSize: number;
  batchCursor = 0;
  lastBatchGradient: number[] = [];
  private latest: TrainingSnapshot;

  constructor(datasetId: DatasetId, width = 8, seed = 2026, options: TrainingOptions = {}) {
    this.dataset = getDataset(datasetId);
    const random = createRandom(seed);
    const outputActivation = this.dataset.task === "classification" ? [new Linear(width, this.dataset.outputFeatures, random, "output")] : [new Linear(width, 1, random, "output")];
    this.model = new Sequential([
      new Linear(this.dataset.inputFeatures, width, random, "input"),
      new Tanh(),
      new Linear(width, width, random, "hidden"),
      new Tanh(),
      ...outputActivation,
    ]);
    const learningRate = datasetId === "sine" ? 0.08 : datasetId === "spiral" ? 0.28 : 0.35;
    this.batchSize = Math.min(Math.max(1, options.batchSize ?? this.dataset.inputs.length), this.dataset.inputs.length);
    const optimizer = options.optimizer ?? "momentum";
    this.optimizer = optimizer === "adam" ? new Adam(this.model.parameters(), datasetId === "sine" ? 0.015 : 0.035) : new SGD(this.model.parameters(), learningRate, optimizer === "momentum" ? 0.82 : 0);
    this.latest = this.evaluate();
  }

  setBatchSize(size: number): void {
    this.batchSize = Math.min(Math.max(1, Math.round(size)), this.dataset.inputs.length);
    this.batchCursor = 0;
  }

  private batchIndices(): number[] {
    const count = this.dataset.inputs.length;
    const indices = Array.from({ length: this.batchSize }, (_, offset) => (this.batchCursor + offset) % count);
    this.batchCursor = (this.batchCursor + this.batchSize) % count;
    return indices;
  }

  private inputTensor(indices = this.dataset.inputs.map((_value, index) => index)): Tensor {
    return Tensor.from(flatten(indices.map((index) => this.dataset.inputs[index])), [indices.length, this.dataset.inputFeatures]);
  }

  private targetTensor(indices = this.dataset.targets.map((_value, index) => index)): Tensor {
    return Tensor.from(flatten(indices.map((index) => this.dataset.targets[index])), [indices.length, this.dataset.outputFeatures]);
  }

  private score(output: number[]): number | undefined {
    if (this.dataset.task !== "classification") return undefined;
    const classes = this.dataset.outputFeatures;
    let correct = 0;
    for (let row = 0; row < this.dataset.labels.length; row += 1) {
      const prediction = output.slice(row * classes, (row + 1) * classes).reduce((best, value, index, values) => value > values[best] ? index : best, 0);
      if (prediction === this.dataset.labels[row]) correct += 1;
    }
    return correct / this.dataset.labels.length;
  }

  evaluate(): TrainingSnapshot {
    return Tensor.noGrad(() => {
      const prediction = this.model.forward(this.inputTensor());
      const loss = this.dataset.task === "classification" ? crossEntropy(prediction, this.targetTensor()) : meanSquaredError(prediction, this.targetTensor());
      const output = this.dataset.task === "classification" ? prediction.softmax().toArray() : prediction.toArray();
      return { epoch: this.epoch, loss: loss.item(), accuracy: this.score(output), output, gradientMagnitude: 0 };
    });
  }

  trainOne(): TrainingSnapshot {
    this.optimizer.zeroGrad();
    const indices = this.batchIndices();
    const prediction = this.model.forward(this.inputTensor(indices));
    const loss = this.dataset.task === "classification" ? crossEntropy(prediction, this.targetTensor(indices)) : meanSquaredError(prediction, this.targetTensor(indices));
    loss.backward();
    const gradients = this.model.parameters().flatMap((parameter) => parameter.grad);
    const gradientMagnitude = Math.sqrt(gradients.reduce((sum, gradient) => sum + gradient * gradient, 0) / gradients.length);
    this.lastBatchGradient = gradients.slice(0, Math.min(8, gradients.length));
    this.optimizer.step();
    this.epoch += 1;
    this.latest = { ...this.evaluate(), gradientMagnitude };
    this.history.push(this.latest.loss);
    return this.latest;
  }

  train(epochs: number): TrainingSnapshot {
    for (let index = 0; index < epochs; index += 1) this.trainOne();
    return this.latest;
  }

  snapshot(): TrainingSnapshot {
    return this.latest;
  }

  predict(points: number[][]): number[] {
    return Tensor.noGrad(() => {
      const output = this.model.forward(Tensor.from(flatten(points), [points.length, this.dataset.inputFeatures]));
      return this.dataset.task === "classification" ? output.softmax().toArray() : output.toArray();
    });
  }

  previewPerturbation(parameterIndex: number, delta: number): { before: TrainingSnapshot; after: TrainingSnapshot; parameter: string } {
    const parameters = this.model.parameters();
    const parameter = parameters[Math.min(Math.max(0, parameterIndex), parameters.length - 1)];
    const before = this.evaluate();
    parameter.data[0] += delta;
    const after = this.evaluate();
    parameter.data[0] -= delta;
    return { before, after, parameter: parameter.label ?? `parameter-${parameterIndex}` };
  }
}
