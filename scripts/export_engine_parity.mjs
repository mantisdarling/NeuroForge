import { writeFileSync } from "node:fs";
import { Tensor } from "../client/src/engine/tensor.ts";
import { crossEntropy } from "../client/src/engine/nn.ts";

const input = Tensor.from([0.2, -0.4, 0.7, 0.1], [2, 2], true, "input");
const weight1 = Tensor.from([0.3, -0.2, 0.15, 0.4, -0.35, 0.25], [2, 3], true, "weight1");
const bias1 = Tensor.from([0.05, -0.1, 0.2], [3], true, "bias1");
const weight2 = Tensor.from([0.2, -0.45, 0.3, 0.15, -0.25, 0.35], [3, 2], true, "weight2");
const bias2 = Tensor.from([0.03, -0.06], [2], true, "bias2");
const target = Tensor.from([1, 0, 0, 1], [2, 2], false, "target");
const hidden = input.matmul(weight1).add(bias1).tanh();
const logits = hidden.matmul(weight2).add(bias2);
const loss = crossEntropy(logits, target);
loss.backward();

writeFileSync("verification/engine-parity.json", JSON.stringify({
  input: input.data, weight1: weight1.data, bias1: bias1.data, weight2: weight2.data, bias2: bias2.data, target: target.data,
  output: logits.data, loss: loss.item(), gradients: { input: input.grad, weight1: weight1.grad, bias1: bias1.grad, weight2: weight2.grad, bias2: bias2.grad },
}, null, 2));
