import { describe, expect, it } from "vitest";
import { DigitLabSession, ShapeCnnSession, compareOptimizers } from "../client/src/engine/advanced";

describe("bounded advanced learning studies", () => {
  it("reduces loss on the browser-local MNIST subset", () => {
    const session = new DigitLabSession();
    const initial = session.train(0);
    const trained = session.train(42);
    expect(trained.loss).toBeLessThan(initial.loss);
    expect(trained.accuracy).toBeGreaterThan(0.55);
  });

  it("learns the generated-shape CNN study through Conv2D and pooling", () => {
    const session = new ShapeCnnSession();
    const trained = session.train(45);
    expect(trained.loss).toBeLessThan(0.8);
    expect(trained.accuracy).toBeGreaterThan(0.8);
  });

  it("returns three finite traces from matched optimizer initializations", () => {
    const traces = compareOptimizers("xor", 8, 24);
    expect(traces).toHaveLength(3);
    traces.forEach((trace) => {
      expect(trace.history).toHaveLength(24);
      expect(Number.isFinite(trace.finalLoss)).toBe(true);
    });
  });
});
