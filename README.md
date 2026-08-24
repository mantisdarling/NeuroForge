# NeuroForge

**NeuroForge** is a browser-native TypeScript laboratory for seeing how a neural network learns. It implements reverse-mode automatic differentiation, small dense networks, optimization, and an interactive computation-graph visualizer **without using a machine-learning runtime**.

> “Trace the gradient, not the mystery.”

Automatic differentiation records how values depend on prior operations, then applies the chain rule backward through that computation graph. Backpropagation is the name commonly used for this reverse traversal when differentiating a scalar learning objective.[1] [2] NeuroForge exposes the process rather than hiding it behind a framework API.

| Area | Included implementation |
| --- | --- |
| Core math | Elementwise add, subtract, multiply, divide, power, exponential, logarithm, sum, mean, transpose, matrix multiplication, ReLU, sigmoid, tanh, and stable softmax. |
| Reverse mode | A topological graph traversal with local backward rules and accumulated gradients at shared parents. |
| Neural network library | `Linear`, `Tanh`, `ReLU`, `Sigmoid`, `Sequential`, mean-squared error, cross-entropy, and SGD with momentum. |
| Live playground | XOR, two-class spiral, and sine regression; hidden-width control; bounded browser-side training; loss trace; decision or regression field; computation-graph state. |
| Quality gates | Finite-difference gradient checks, deterministic convergence tests, strict TypeScript checks, production build, dependency audit, and credential-marker scan. |

## Explore the visualizer

Run the app and choose a dataset in the operations rail. **Observe 320 epochs** performs a bounded local training session and updates the loss trace, prediction field, node readouts, and gradient magnitude. **Trace next backward pass** executes one real optimization step and moves the active signal through the graph. Every training calculation is local to the browser; there is no model upload, API call, account, or persistence layer.

The graph canvas is the primary observation surface. It shows the forward computation from input to loss, while teal pulses represent reverse gradient flow. On narrow screens, swipe horizontally across the canvas to trace every operation; dataset selection and network-width controls remain available above the action controls.

| Dataset | Task | Target behavior |
| --- | --- | --- |
| XOR | Binary classification | Learns a nonlinear decision boundary. |
| Spiral | Binary classification | Separates two interleaved radial classes. |
| Sine | Regression | Approximates a sampled sine curve. |

## How reverse-mode autograd works

`Tensor` owns flat numeric data, shape, a gradient buffer, references to its parent tensors, and the operation that created it. Each differentiable operation constructs an output tensor and closes over a **local backward rule**. The forward pass builds a directed acyclic computation graph; it does not eagerly apply the full chain rule.

Calling `backward()` first creates a topological ordering from the output back through its parents. It then seeds the scalar output gradient with `1` and runs the stored local rules in reverse topological order. A rule adds its contribution into each parent’s gradient buffer. This accumulation is important whenever a value feeds more than one child: reverse mode sums every child-path contribution, exactly as required by the multivariable chain rule.[2]

```text
forward:  x ──► linear ──► tanh ──► linear ──► loss
                                 
reverse:  ∂L/∂x ◄── local rules ◄── ∂L/∂loss = 1
```

The `softmax()` implementation subtracts the per-row maximum before exponentiation, which keeps the calculation numerically stable for the small browser demonstrations. Cross-entropy consumes a one-hot target and averages the objective across the batch.

## Correctness verification

Gradient checking compares a derivative produced by reverse mode with a central finite-difference approximation:

```text
f′(x) ≈ (f(x + ε) − f(x − ε)) / (2ε)
```

This repository checks a composed scalar expression, matrix multiplication followed by `tanh`, and a softmax-log loss. The convergence suite also verifies that XOR reaches full training accuracy and sine regression drives loss below the defined acceptance threshold.

| Command | Result from the final local verification |
| --- | --- |
| `pnpm check` | Passed with strict TypeScript checking. |
| `pnpm test` | Passed: 5 tests across 2 test files. The three finite-difference gradient checks and two deterministic training tests passed. |
| Browser XOR run | 320 epochs; loss decreased from `0.70864` to `0.00058`; accuracy reached `100%`. |
| `pnpm security:scan` | Passed; no credential markers were found in 15 scanned files. |
| `pnpm build` | Passed; production Vite bundle produced successfully. |
| `pnpm audit` | Passed; no known dependency vulnerabilities reported. |

Run the suite yourself with:

```bash
pnpm install --frozen-lockfile
pnpm check
pnpm test
pnpm security:scan
pnpm build
pnpm audit
```

## Project structure

```text
client/src/engine/tensor.ts      Reverse-mode tensor and differentiable operations
client/src/engine/nn.ts          Layers, losses, seeded initialization, and SGD
client/src/engine/training.ts    Deterministic toy datasets and local training session
client/src/components/           SVG graph, loss trace, and model-field views
tests/                           Gradient checks and convergence tests
scripts/scan-secrets.mjs         Narrow credential-marker scan
.github/workflows/ci.yml         Push and pull-request verification gates
vercel.json                      Static-hosting security headers
```

## Security posture

NeuroForge is intentionally a **static, client-only application**. It does not process credentials, accept free-form executable input, fetch third-party data at runtime, persist user data, or expose server-side mutation endpoints. This sharply limits the application attack surface, but it is not a claim that no website can ever be attacked.

The repository applies a small set of defense-in-depth controls. Vercel configuration supplies a restrictive content security policy, frame denial, MIME-sniffing protection, referrer controls, and a minimal permissions policy. Continuous integration installs from the lockfile, runs the credential-marker scan, type checks, tests, builds, and fails the job when the production dependency audit reports high-severity issues. Keep the deployment account protected with multi-factor authentication and review dependency updates before merging them.

| Boundary | Implemented control | Limitation |
| --- | --- | --- |
| Source repository | `.gitignore`, credential-marker scan, locked dependency installation, and CI gates. | A secret accidentally committed before scanning must still be rotated outside this repository. |
| Browser document | CSP, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, referrer policy, and permissions policy. | Hosting-provider account, DNS, TLS, and availability controls remain provider responsibilities. |
| Training runtime | Fixed datasets, bounded width slider, deterministic seeded data, no remote model execution. | This is an educational in-browser engine, not a sandbox for arbitrary untrusted code. |

## Deployment

The repository is prepared for static deployment on Vercel. Import `mantisdarling/NeuroForge`, retain the default build command `pnpm build`, and publish the generated `dist` directory. The included `vercel.json` applies the browser-facing security headers at deployment time.

## References

[1] [Baydin, Pearlmutter, Radul, and Siskind, “Automatic Differentiation in Machine Learning: a Survey,” *Journal of Machine Learning Research*, 2018.](http://www.jmlr.org/papers/v18/17-468.html)

[2] [Auto-eD, “Module 3: The Reverse Mode of Automatic Differentiation.”](https://auto-ed.readthedocs.io/en/latest/mod3.html)

[3] [Dive into Deep Learning, “Automatic Differentiation.”](https://d2l.ai/chapter_preliminaries/autograd.html)
