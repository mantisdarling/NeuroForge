import { describe, expect, it } from "vitest";
import { TrainingSession } from "../client/src/engine/training";

describe("training examples", () => {
  it("learns XOR with decreasing loss and full accuracy", () => {
    const session = new TrainingSession("xor", 8, 2026);
    const initial = session.snapshot();
    const final = session.train(500);
    expect(final.loss).toBeLessThan(initial.loss * 0.12);
    expect(final.accuracy).toBe(1);
  });

  it("reduces sine regression loss", () => {
    const session = new TrainingSession("sine", 12, 2027);
    const initial = session.snapshot();
    const final = session.train(700);
    expect(final.loss).toBeLessThan(initial.loss * 0.16);
    expect(final.loss).toBeLessThan(0.03);
  });
});

