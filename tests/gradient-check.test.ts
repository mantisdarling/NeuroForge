import { describe, expect, it } from "vitest";
import { Tensor } from "../client/src/engine/tensor";

const EPSILON = 1e-5;

function finiteDifference(f: (x: number) => number, x: number): number {
  return (f(x + EPSILON) - f(x - EPSILON)) / (2 * EPSILON);
}

describe("automatic differentiation gradient checks", () => {
  it("matches finite differences for a composed scalar expression", () => {
    const objective = (xValue: number): { loss: Tensor; x: Tensor } => {
      const x = Tensor.scalar(xValue, true, "x");
      const y = x.mul(1.7).add(0.2).pow(2).tanh().exp().log();
      return { loss: y, x };
    };
    const analytical = objective(0.37);
    analytical.loss.backward();
    const numerical = finiteDifference((value) => objective(value).loss.item(), 0.37);
    expect(Math.abs(analytical.x.grad[0] - numerical)).toBeLessThan(1e-5);
  });

  it("matches finite differences through matrix multiplication", () => {
    const objective = (value: number): { loss: Tensor; first: Tensor } => {
      const first = Tensor.from([value, -0.2, 0.4, 0.1], [2, 2], true, "first");
      const second = Tensor.from([0.3, -0.7, 0.5, 0.2], [2, 2], true, "second");
      return { loss: first.matmul(second).tanh().sum(), first };
    };
    const analytical = objective(0.25);
    analytical.loss.backward();
    const numerical = finiteDifference((value) => objective(value).loss.item(), 0.25);
    expect(Math.abs(analytical.first.grad[0] - numerical)).toBeLessThan(1e-5);
  });

  it("matches finite differences through softmax and log", () => {
    const objective = (value: number): { loss: Tensor; logits: Tensor } => {
      const logits = Tensor.from([value, -0.4, 0.15], [1, 3], true, "logits");
      const target = Tensor.from([1, 0, 0], [1, 3]);
      return { loss: logits.softmax().log().mul(target).sum().neg(), logits };
    };
    const analytical = objective(0.2);
    analytical.loss.backward();
    const numerical = finiteDifference((value) => objective(value).loss.item(), 0.2);
    expect(Math.abs(analytical.logits.grad[0] - numerical)).toBeLessThan(1e-5);
  });
});

