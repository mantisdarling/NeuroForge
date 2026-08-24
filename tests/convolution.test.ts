import { describe, expect, it } from "vitest";
import { Tensor } from "../client/src/engine/tensor";

const EPSILON = 1e-5;
const finiteDifference = (fn: (value: number) => number, value: number) => (fn(value + EPSILON) - fn(value - EPSILON)) / (2 * EPSILON);

describe("convolutional reverse-mode checks", () => {
  it("matches finite differences for a valid Conv2D kernel weight", () => {
    const objective = (weightValue: number) => {
      const input = Tensor.from([0.1, -0.2, 0.3, 0.4, 0.5, -0.1, -0.3, 0.2, 0.6], [1, 1, 3, 3], true, "input");
      const kernel = Tensor.from([weightValue, 0.2, -0.3, 0.4], [1, 1, 2, 2], true, "kernel");
      return { loss: input.conv2d(kernel).tanh().sum(), kernel };
    };
    const analytical = objective(0.25);
    analytical.loss.backward();
    const numerical = finiteDifference((value) => objective(value).loss.item(), 0.25);
    expect(Math.abs(analytical.kernel.grad[0] - numerical)).toBeLessThan(1e-5);
  });

  it("routes a max-pooling gradient to the winning cell", () => {
    const input = Tensor.from([0.2, 0.9, 0.5, 0.1], [1, 1, 2, 2], true, "pool-input");
    const output = input.maxPool2d(2).sum();
    output.backward();
    expect(input.grad).toEqual([0, 1, 0, 0]);
  });
});
