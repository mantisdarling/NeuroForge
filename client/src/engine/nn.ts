import { Tensor } from "./tensor";

export interface Layer {
  forward(input: Tensor): Tensor;
  parameters(): Tensor[];
}

export type Random = () => number;

export function createRandom(seed = 2026): Random {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export class Linear implements Layer {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(inputFeatures: number, outputFeatures: number, random: Random, label = "linear") {
    const limit = Math.sqrt(6 / (inputFeatures + outputFeatures));
    const weights = Array.from({ length: inputFeatures * outputFeatures }, () => (random() * 2 - 1) * limit);
    this.weight = Tensor.from(weights, [inputFeatures, outputFeatures], true, `${label}.weight`);
    this.bias = Tensor.zeros([outputFeatures], true, `${label}.bias`);
  }

  forward(input: Tensor): Tensor {
    return input.matmul(this.weight).add(this.bias);
  }

  parameters(): Tensor[] {
    return [this.weight, this.bias];
  }
}

export class Conv2D implements Layer {
  readonly weight: Tensor;
  readonly bias: Tensor;

  constructor(inputChannels: number, outputChannels: number, kernelSize: number, random: Random, label = "conv") {
    const scale = Math.sqrt(2 / (inputChannels * kernelSize * kernelSize));
    this.weight = Tensor.from(Array.from({ length: outputChannels * inputChannels * kernelSize * kernelSize }, () => (random() * 2 - 1) * scale), [outputChannels, inputChannels, kernelSize, kernelSize], true, `${label}.weight`);
    this.bias = Tensor.zeros([outputChannels], true, `${label}.bias`);
  }

  forward(input: Tensor): Tensor { return input.conv2d(this.weight, this.bias); }
  parameters(): Tensor[] { return [this.weight, this.bias]; }
}

export class MaxPool2D implements Layer {
  constructor(private readonly window = 2) {}
  forward(input: Tensor): Tensor { return input.maxPool2d(this.window); }
  parameters(): Tensor[] { return []; }
}

export class ReLU implements Layer {
  forward(input: Tensor): Tensor { return input.relu(); }
  parameters(): Tensor[] { return []; }
}

export class Sigmoid implements Layer {
  forward(input: Tensor): Tensor { return input.sigmoid(); }
  parameters(): Tensor[] { return []; }
}

export class Tanh implements Layer {
  forward(input: Tensor): Tensor { return input.tanh(); }
  parameters(): Tensor[] { return []; }
}

export class Sequential implements Layer {
  constructor(readonly layers: Layer[]) {}

  forward(input: Tensor): Tensor {
    return this.layers.reduce((value, layer) => layer.forward(value), input);
  }

  parameters(): Tensor[] {
    return this.layers.flatMap((layer) => layer.parameters());
  }
}

export class SGD {
  private readonly velocity = new Map<Tensor, number[]>();

  constructor(readonly parameters: Tensor[], readonly learningRate: number, readonly momentum = 0) {
    if (learningRate <= 0 || learningRate > 1) throw new Error("SGD learningRate must be in (0, 1].");
    if (momentum < 0 || momentum >= 1) throw new Error("SGD momentum must be in [0, 1).");
  }

  zeroGrad(): void {
    this.parameters.forEach((parameter) => parameter.zeroGrad());
  }

  step(): void {
    this.parameters.forEach((parameter) => {
      const previous = this.velocity.get(parameter) ?? Array(parameter.size).fill(0);
      const next = parameter.grad.map((gradient, index) => this.momentum * previous[index] + gradient);
      this.velocity.set(parameter, next);
      parameter.data.forEach((_value, index) => { parameter.data[index] -= this.learningRate * next[index]; });
    });
  }
}

export class Adam {
  private readonly firstMoment = new Map<Tensor, number[]>();
  private readonly secondMoment = new Map<Tensor, number[]>();
  private stepCount = 0;

  constructor(readonly parameters: Tensor[], readonly learningRate = 0.002, private readonly beta1 = 0.9, private readonly beta2 = 0.999, private readonly epsilon = 1e-8) {
    if (learningRate <= 0 || learningRate > 1) throw new Error("Adam learningRate must be in (0, 1].");
  }

  zeroGrad(): void { this.parameters.forEach((parameter) => parameter.zeroGrad()); }

  step(): void {
    this.stepCount += 1;
    this.parameters.forEach((parameter) => {
      const first = this.firstMoment.get(parameter) ?? Array(parameter.size).fill(0);
      const second = this.secondMoment.get(parameter) ?? Array(parameter.size).fill(0);
      parameter.data.forEach((_value, index) => {
        first[index] = this.beta1 * first[index] + (1 - this.beta1) * parameter.grad[index];
        second[index] = this.beta2 * second[index] + (1 - this.beta2) * parameter.grad[index] ** 2;
        const correctedFirst = first[index] / (1 - this.beta1 ** this.stepCount);
        const correctedSecond = second[index] / (1 - this.beta2 ** this.stepCount);
        parameter.data[index] -= this.learningRate * correctedFirst / (Math.sqrt(correctedSecond) + this.epsilon);
      });
      this.firstMoment.set(parameter, first);
      this.secondMoment.set(parameter, second);
    });
  }
}

export function meanSquaredError(prediction: Tensor, target: Tensor): Tensor {
  return prediction.sub(target).pow(2).mean();
}

export function crossEntropy(logits: Tensor, oneHotTarget: Tensor): Tensor {
  if (logits.shape.length !== 2 || oneHotTarget.shape.length !== 2) throw new Error("crossEntropy() expects rank-2 logits and targets.");
  if (logits.shape[0] !== oneHotTarget.shape[0] || logits.shape[1] !== oneHotTarget.shape[1]) throw new Error("crossEntropy() target shape must match logits.");
  return logits.softmax().log().mul(oneHotTarget).sum().mul(-1 / logits.shape[0]);
}
