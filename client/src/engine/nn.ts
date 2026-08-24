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

export function meanSquaredError(prediction: Tensor, target: Tensor): Tensor {
  return prediction.sub(target).pow(2).mean();
}

export function crossEntropy(logits: Tensor, oneHotTarget: Tensor): Tensor {
  if (logits.shape.length !== 2 || oneHotTarget.shape.length !== 2) throw new Error("crossEntropy() expects rank-2 logits and targets.");
  if (logits.shape[0] !== oneHotTarget.shape[0] || logits.shape[1] !== oneHotTarget.shape[1]) throw new Error("crossEntropy() target shape must match logits.");
  return logits.softmax().log().mul(oneHotTarget).sum().mul(-1 / logits.shape[0]);
}

