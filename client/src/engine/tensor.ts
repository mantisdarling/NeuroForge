export type Shape = readonly number[];

type TensorOptions = {
  requiresGrad?: boolean;
  label?: string;
  operation?: string;
};

let nextTensorId = 1;

function sizeOf(shape: Shape): number {
  return shape.reduce((size, dimension) => size * dimension, 1);
}

function stridesFor(shape: Shape): number[] {
  const strides = Array(shape.length).fill(1);
  for (let index = shape.length - 2; index >= 0; index -= 1) {
    strides[index] = strides[index + 1] * shape[index + 1];
  }
  return strides;
}

function coordinatesOf(index: number, shape: Shape): number[] {
  const strides = stridesFor(shape);
  return shape.map((dimension, axis) => Math.floor(index / strides[axis]) % dimension);
}

function offsetForBroadcast(outputIndex: number, outputShape: Shape, inputShape: Shape): number {
  const outputCoordinates = coordinatesOf(outputIndex, outputShape);
  const inputStrides = stridesFor(inputShape);
  const leadingAxes = outputShape.length - inputShape.length;
  return inputShape.reduce((offset, dimension, axis) => {
    const coordinate = dimension === 1 ? 0 : outputCoordinates[axis + leadingAxes];
    return offset + coordinate * inputStrides[axis];
  }, 0);
}

function broadcastShape(left: Shape, right: Shape): number[] {
  const result: number[] = [];
  const dimensions = Math.max(left.length, right.length);
  for (let axis = 0; axis < dimensions; axis += 1) {
    const leftDimension = left[left.length - 1 - axis] ?? 1;
    const rightDimension = right[right.length - 1 - axis] ?? 1;
    if (leftDimension !== rightDimension && leftDimension !== 1 && rightDimension !== 1) {
      throw new Error(`Cannot broadcast [${left}] with [${right}].`);
    }
    result.unshift(Math.max(leftDimension, rightDimension));
  }
  return result;
}

function reduceBroadcastGradient(gradient: number[], outputShape: Shape, targetShape: Shape): number[] {
  const reduced = Array(sizeOf(targetShape)).fill(0);
  for (let outputIndex = 0; outputIndex < gradient.length; outputIndex += 1) {
    reduced[offsetForBroadcast(outputIndex, outputShape, targetShape)] += gradient[outputIndex];
  }
  return reduced;
}

function addInto(target: number[], source: number[]): void {
  for (let index = 0; index < target.length; index += 1) target[index] += source[index];
}

function ensureSameShape(left: Shape, right: Shape, operation: string): void {
  if (left.length !== right.length || left.some((dimension, index) => dimension !== right[index])) {
    throw new Error(`${operation} requires matching shapes, received [${left}] and [${right}].`);
  }
}

/**
 * A compact reverse-mode automatic-differentiation tensor. Every operation records
 * its parents and a local backward rule; backward() traverses the graph topologically.
 */
export class Tensor {
  private static gradientsEnabled = true;

  readonly id = nextTensorId++;
  readonly data: number[];
  readonly shape: number[];
  readonly requiresGrad: boolean;
  readonly parents: Tensor[];
  readonly label?: string;
  readonly operation?: string;
  grad: number[];
  private backwardRule: () => void = () => undefined;

  constructor(data: number[], shape: Shape = [], options: TensorOptions = {}, parents: Tensor[] = []) {
    if (!shape.every((dimension) => Number.isInteger(dimension) && dimension > 0)) {
      throw new Error(`Tensor shape must contain positive integers, received [${shape}].`);
    }
    if (data.length !== sizeOf(shape)) {
      throw new Error(`Tensor data length ${data.length} does not match shape [${shape}].`);
    }
    if (!data.every(Number.isFinite)) throw new Error("Tensor data must contain finite numbers.");

    this.data = data;
    this.shape = [...shape];
    this.requiresGrad = Boolean(options.requiresGrad) && Tensor.gradientsEnabled;
    this.parents = parents;
    this.label = options.label;
    this.operation = options.operation;
    this.grad = Array(data.length).fill(0);
  }

  static scalar(value: number, requiresGrad = false, label?: string): Tensor {
    return new Tensor([value], [], { requiresGrad, label });
  }

  static from(data: number[], shape: Shape, requiresGrad = false, label?: string): Tensor {
    return new Tensor([...data], shape, { requiresGrad, label });
  }

  static zeros(shape: Shape, requiresGrad = false, label?: string): Tensor {
    return new Tensor(Array(sizeOf(shape)).fill(0), shape, { requiresGrad, label });
  }

  static noGrad<T>(callback: () => T): T {
    const previous = Tensor.gradientsEnabled;
    Tensor.gradientsEnabled = false;
    try {
      return callback();
    } finally {
      Tensor.gradientsEnabled = previous;
    }
  }

  get size(): number {
    return this.data.length;
  }

  item(): number {
    if (this.size !== 1) throw new Error("item() is only valid for scalar tensors.");
    return this.data[0];
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  private operationResult(data: number[], shape: Shape, parents: Tensor[], operation: string): Tensor {
    return new Tensor(data, shape, {
      requiresGrad: parents.some((parent) => parent.requiresGrad),
      operation,
    }, parents);
  }

  private elementwise(
    input: Tensor | number,
    operation: string,
    forward: (left: number, right: number) => number,
    leftGradient: (upstream: number, left: number, right: number) => number,
    rightGradient: (upstream: number, left: number, right: number) => number,
  ): Tensor {
    const right = input instanceof Tensor ? input : Tensor.scalar(input);
    const shape = broadcastShape(this.shape, right.shape);
    const data = Array.from({ length: sizeOf(shape) }, (_, index) => {
      const leftValue = this.data[offsetForBroadcast(index, shape, this.shape)];
      const rightValue = right.data[offsetForBroadcast(index, shape, right.shape)];
      return forward(leftValue, rightValue);
    });
    const output = this.operationResult(data, shape, [this, right], operation);
    if (output.requiresGrad) {
      output.backwardRule = () => {
        if (this.requiresGrad) {
          const gradient = output.grad.map((upstream, index) => {
            const left = this.data[offsetForBroadcast(index, shape, this.shape)];
            const rightValue = right.data[offsetForBroadcast(index, shape, right.shape)];
            return leftGradient(upstream, left, rightValue);
          });
          addInto(this.grad, reduceBroadcastGradient(gradient, shape, this.shape));
        }
        if (right.requiresGrad) {
          const gradient = output.grad.map((upstream, index) => {
            const left = this.data[offsetForBroadcast(index, shape, this.shape)];
            const rightValue = right.data[offsetForBroadcast(index, shape, right.shape)];
            return rightGradient(upstream, left, rightValue);
          });
          addInto(right.grad, reduceBroadcastGradient(gradient, shape, right.shape));
        }
      };
    }
    return output;
  }

  add(input: Tensor | number): Tensor {
    return this.elementwise(input, "add", (left, right) => left + right, (gradient) => gradient, (gradient) => gradient);
  }

  sub(input: Tensor | number): Tensor {
    return this.elementwise(input, "sub", (left, right) => left - right, (gradient) => gradient, (gradient) => -gradient);
  }

  mul(input: Tensor | number): Tensor {
    return this.elementwise(input, "mul", (left, right) => left * right, (gradient, _left, right) => gradient * right, (gradient, left) => gradient * left);
  }

  div(input: Tensor | number): Tensor {
    return this.elementwise(input, "div", (left, right) => left / right, (gradient, _left, right) => gradient / right, (gradient, left, right) => -gradient * left / (right * right));
  }

  pow(input: Tensor | number): Tensor {
    return this.elementwise(
      input,
      "pow",
      (left, right) => left ** right,
      (gradient, left, right) => gradient * right * left ** (right - 1),
      (gradient, left, right) => (left > 0 ? gradient * left ** right * Math.log(left) : 0),
    );
  }

  neg(): Tensor {
    return this.mul(-1);
  }

  exp(): Tensor {
    const output = this.operationResult(this.data.map(Math.exp), this.shape, [this], "exp");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad.map((gradient, index) => gradient * output.data[index]));
    return output;
  }

  log(): Tensor {
    const output = this.operationResult(this.data.map((value) => Math.log(Math.max(value, 1e-12))), this.shape, [this], "log");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad.map((gradient, index) => gradient / Math.max(this.data[index], 1e-12)));
    return output;
  }

  relu(): Tensor {
    const output = this.operationResult(this.data.map((value) => Math.max(0, value)), this.shape, [this], "relu");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad.map((gradient, index) => (this.data[index] > 0 ? gradient : 0)));
    return output;
  }

  sigmoid(): Tensor {
    const output = this.operationResult(this.data.map((value) => 1 / (1 + Math.exp(-value))), this.shape, [this], "sigmoid");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad.map((gradient, index) => gradient * output.data[index] * (1 - output.data[index])));
    return output;
  }

  tanh(): Tensor {
    const output = this.operationResult(this.data.map(Math.tanh), this.shape, [this], "tanh");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad.map((gradient, index) => gradient * (1 - output.data[index] ** 2)));
    return output;
  }

  sum(): Tensor {
    const output = this.operationResult([this.data.reduce((sum, value) => sum + value, 0)], [], [this], "sum");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, this.grad.map(() => output.grad[0]));
    return output;
  }

  mean(): Tensor {
    return this.sum().div(this.size);
  }

  transpose(): Tensor {
    if (this.shape.length !== 2) throw new Error("transpose() currently supports rank-2 tensors.");
    const [rows, columns] = this.shape;
    const data = Array.from({ length: this.size }, (_, index) => {
      const row = index % rows;
      const column = Math.floor(index / rows);
      return this.data[row * columns + column];
    });
    const output = this.operationResult(data, [columns, rows], [this], "transpose");
    if (output.requiresGrad) {
      output.backwardRule = () => {
        for (let row = 0; row < rows; row += 1) {
          for (let column = 0; column < columns; column += 1) this.grad[row * columns + column] += output.grad[column * rows + row];
        }
      };
    }
    return output;
  }

  matmul(input: Tensor): Tensor {
    if (this.shape.length !== 2 || input.shape.length !== 2) throw new Error("matmul() requires two rank-2 tensors.");
    const [rows, inner] = this.shape;
    const [otherInner, columns] = input.shape;
    if (inner !== otherInner) throw new Error(`matmul() dimension mismatch: [${this.shape}] × [${input.shape}].`);
    const data = Array(rows * columns).fill(0);
    for (let row = 0; row < rows; row += 1) {
      for (let column = 0; column < columns; column += 1) {
        for (let index = 0; index < inner; index += 1) data[row * columns + column] += this.data[row * inner + index] * input.data[index * columns + column];
      }
    }
    const output = this.operationResult(data, [rows, columns], [this, input], "matmul");
    if (output.requiresGrad) {
      output.backwardRule = () => {
        if (this.requiresGrad) {
          for (let row = 0; row < rows; row += 1) {
            for (let index = 0; index < inner; index += 1) {
              for (let column = 0; column < columns; column += 1) this.grad[row * inner + index] += output.grad[row * columns + column] * input.data[index * columns + column];
            }
          }
        }
        if (input.requiresGrad) {
          for (let index = 0; index < inner; index += 1) {
            for (let column = 0; column < columns; column += 1) {
              for (let row = 0; row < rows; row += 1) input.grad[index * columns + column] += this.data[row * inner + index] * output.grad[row * columns + column];
            }
          }
        }
      };
    }
    return output;
  }

  flatten(startDimension = 1): Tensor {
    if (startDimension < 0 || startDimension >= this.shape.length) throw new Error("flatten() startDimension must be a valid tensor axis.");
    const leading = this.shape.slice(0, startDimension);
    const flattened = this.shape.slice(startDimension).reduce((product, dimension) => product * dimension, 1);
    const output = this.operationResult([...this.data], [...leading, flattened], [this], "flatten");
    if (output.requiresGrad) output.backwardRule = () => addInto(this.grad, output.grad);
    return output;
  }

  conv2d(kernel: Tensor, bias?: Tensor): Tensor {
    if (this.shape.length !== 4 || kernel.shape.length !== 4) throw new Error("conv2d() expects input [batch, channels, height, width] and kernel [filters, channels, height, width].");
    const [batch, channels, inputHeight, inputWidth] = this.shape;
    const [filters, kernelChannels, kernelHeight, kernelWidth] = kernel.shape;
    if (channels !== kernelChannels) throw new Error("conv2d() channel count must match kernel channels.");
    if (bias && (bias.shape.length !== 1 || bias.shape[0] !== filters)) throw new Error("conv2d() bias must have one entry per filter.");
    const outputHeight = inputHeight - kernelHeight + 1;
    const outputWidth = inputWidth - kernelWidth + 1;
    if (outputHeight <= 0 || outputWidth <= 0) throw new Error("conv2d() kernel must fit within the input.");
    const inputOffset = (batchIndex: number, channel: number, row: number, column: number) => ((batchIndex * channels + channel) * inputHeight + row) * inputWidth + column;
    const kernelOffset = (filter: number, channel: number, row: number, column: number) => ((filter * channels + channel) * kernelHeight + row) * kernelWidth + column;
    const outputOffset = (batchIndex: number, filter: number, row: number, column: number) => ((batchIndex * filters + filter) * outputHeight + row) * outputWidth + column;
    const data = Array(batch * filters * outputHeight * outputWidth).fill(0);
    for (let sample = 0; sample < batch; sample += 1) for (let filter = 0; filter < filters; filter += 1) for (let row = 0; row < outputHeight; row += 1) for (let column = 0; column < outputWidth; column += 1) {
      let value = bias ? bias.data[filter] : 0;
      for (let channel = 0; channel < channels; channel += 1) for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) for (let kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn += 1) value += this.data[inputOffset(sample, channel, row + kernelRow, column + kernelColumn)] * kernel.data[kernelOffset(filter, channel, kernelRow, kernelColumn)];
      data[outputOffset(sample, filter, row, column)] = value;
    }
    const parents = bias ? [this, kernel, bias] : [this, kernel];
    const output = this.operationResult(data, [batch, filters, outputHeight, outputWidth], parents, "conv2d");
    if (output.requiresGrad) output.backwardRule = () => {
      for (let sample = 0; sample < batch; sample += 1) for (let filter = 0; filter < filters; filter += 1) for (let row = 0; row < outputHeight; row += 1) for (let column = 0; column < outputWidth; column += 1) {
        const upstream = output.grad[outputOffset(sample, filter, row, column)];
        if (bias?.requiresGrad) bias.grad[filter] += upstream;
        for (let channel = 0; channel < channels; channel += 1) for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow += 1) for (let kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn += 1) {
          const sourceIndex = inputOffset(sample, channel, row + kernelRow, column + kernelColumn);
          const kernelIndex = kernelOffset(filter, channel, kernelRow, kernelColumn);
          if (this.requiresGrad) this.grad[sourceIndex] += upstream * kernel.data[kernelIndex];
          if (kernel.requiresGrad) kernel.grad[kernelIndex] += upstream * this.data[sourceIndex];
        }
      }
    };
    return output;
  }

  maxPool2d(window = 2): Tensor {
    if (this.shape.length !== 4 || !Number.isInteger(window) || window < 1) throw new Error("maxPool2d() expects a rank-4 tensor and a positive integer window.");
    const [batch, channels, height, width] = this.shape;
    const outputHeight = Math.floor(height / window);
    const outputWidth = Math.floor(width / window);
    const inputOffset = (sample: number, channel: number, row: number, column: number) => ((sample * channels + channel) * height + row) * width + column;
    const outputOffset = (sample: number, channel: number, row: number, column: number) => ((sample * channels + channel) * outputHeight + row) * outputWidth + column;
    const maxima = Array(batch * channels * outputHeight * outputWidth).fill(0);
    const data = maxima.map((_, outputIndex) => {
      const [sample, channel, row, column] = coordinatesOf(outputIndex, [batch, channels, outputHeight, outputWidth]);
      let bestIndex = inputOffset(sample, channel, row * window, column * window);
      for (let deltaRow = 0; deltaRow < window; deltaRow += 1) for (let deltaColumn = 0; deltaColumn < window; deltaColumn += 1) {
        const candidate = inputOffset(sample, channel, row * window + deltaRow, column * window + deltaColumn);
        if (this.data[candidate] > this.data[bestIndex]) bestIndex = candidate;
      }
      maxima[outputOffset(sample, channel, row, column)] = bestIndex;
      return this.data[bestIndex];
    });
    const output = this.operationResult(data, [batch, channels, outputHeight, outputWidth], [this], "maxPool2d");
    if (output.requiresGrad) output.backwardRule = () => output.grad.forEach((gradient, index) => { this.grad[maxima[index]] += gradient; });
    return output;
  }

  softmax(): Tensor {
    if (this.shape.length > 2) throw new Error("softmax() supports vectors or rank-2 tensors.");
    const classes = this.shape.at(-1) ?? 1;
    const rows = this.size / classes;
    const data = Array(this.size).fill(0);
    for (let row = 0; row < rows; row += 1) {
      const start = row * classes;
      const maximum = Math.max(...this.data.slice(start, start + classes));
      const exponentials = this.data.slice(start, start + classes).map((value) => Math.exp(value - maximum));
      const normalizer = exponentials.reduce((sum, value) => sum + value, 0);
      for (let column = 0; column < classes; column += 1) data[start + column] = exponentials[column] / normalizer;
    }
    const output = this.operationResult(data, this.shape, [this], "softmax");
    if (output.requiresGrad) {
      output.backwardRule = () => {
        for (let row = 0; row < rows; row += 1) {
          const start = row * classes;
          let dot = 0;
          for (let column = 0; column < classes; column += 1) dot += output.grad[start + column] * output.data[start + column];
          for (let column = 0; column < classes; column += 1) this.grad[start + column] += output.data[start + column] * (output.grad[start + column] - dot);
        }
      };
    }
    return output;
  }

  backward(seed?: number[]): void {
    if (seed && seed.length !== this.size) throw new Error("The backward seed must match the output shape.");
    if (!seed && this.size !== 1) throw new Error("backward() requires an explicit seed for non-scalar tensors.");
    this.grad = seed ? [...seed] : [1];
    const order: Tensor[] = [];
    const visited = new Set<number>();
    const visit = (node: Tensor): void => {
      if (visited.has(node.id)) return;
      visited.add(node.id);
      node.parents.forEach(visit);
      order.push(node);
    };
    visit(this);
    for (let index = order.length - 1; index >= 0; index -= 1) order[index].backwardRule();
  }

  toArray(): number[] {
    return [...this.data];
  }
}
