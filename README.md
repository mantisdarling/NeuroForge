# NeuroForge

**NeuroForge** is a browser-native learning instrument for reverse-mode automatic differentiation. It makes neural-network computation observable: values move forward through a graph, gradients return backward through local rules, and every experiment runs locally in the browser with TypeScript rather than a browser machine-learning framework.

> **Trace the gradient, not the mystery.**

**Live application:** [neuroforge-nu.vercel.app](https://neuroforge-nu.vercel.app/)

## Why NeuroForge

Automatic differentiation records the operations used to compute a value, then applies the chain rule in reverse to calculate derivatives of a scalar loss. Backpropagation is the familiar neural-network form of that reverse traversal.[1] [2] NeuroForge turns that process into an interactive visual experience rather than treating it as a black-box framework feature.

| Principle | NeuroForge approach |
| --- | --- |
| **Inspectable** | Shows computation graphs, local derivative rules, loss traces, and decision fields. |
| **Local** | Runs training, sketch analysis, and bounded studies entirely in the visitor’s browser. |
| **From scratch** | Uses a custom TypeScript tensor engine and neural-network primitives, not a browser ML runtime. |
| **Verified** | Combines finite differences, deterministic convergence studies, strict typing, dependency checks, and an independent PyTorch parity benchmark. |

## Interactive features

| Feature | What it demonstrates |
| --- | --- |
| **Live Lab** | Train XOR, spiral classification, or sine regression; inspect loss, predictions, graph values, and reverse gradients. |
| **Guided first visit** | A short route explains dataset choice, training, and backward-pass tracing without blocking the lab. |
| **Computation graph** | Displays a forward route from inputs to loss and a reverse route for gradient accumulation. |
| **Digit field** | Draw a digit on a local canvas and inspect a 10-class distribution from a bounded 14 × 14 study. |
| **Sensitivity field** | Temporarily perturb a visible weight, measure the change, then restore the original training state. |
| **Mini-batch lens** | Compare batch sizes of 1, 2, 4, and 8 while viewing per-example gradients and their mean update. |
| **CNN shapes** | Train a compact valid-convolution, max-pooling, and dense classifier on deterministic geometry samples. |
| **Optimizer evidence** | Compare SGD, momentum, and Adam from an identical XOR initialization. |
| **Accessibility controls** | Includes keyboard focus feedback, keyboard support for the digit canvas, reduced-motion handling, and a persistent high-contrast mode. |

## Architecture

NeuroForge is a **static React and TypeScript application**. There is no authentication system, API server, database, remote model call, upload pipeline, or runtime dataset download. Each browser has its own bounded in-memory training session, which keeps experiments independent and avoids a shared inference or training service.

```text
Input → linear layer → activation → linear layer → loss
  x   →   W₁x+b₁   →  tanh(z)  →  W₂a+b₂  →  L

Backward pass: ∂L/∂θ flows from loss to every parent through local rules
```

The custom engine stores a tensor’s data, gradient buffer, shape, parent references, and local backward rule. Calling `backward()` builds a topological order, seeds the scalar loss gradient, and accumulates every valid contribution while traversing the graph in reverse. This is the core reverse-mode pattern described in automatic-differentiation literature.[1] [2]

| Layer | Included capabilities |
| --- | --- |
| **Tensor engine** | Arithmetic, exponential and logarithmic functions, reductions, transpose, matrix multiplication, stable softmax, valid Conv2D, max pooling, and reverse-mode gradient accumulation. |
| **Neural-network primitives** | `Linear`, `Conv2D`, `MaxPool2D`, `Tanh`, `ReLU`, `Sigmoid`, `Sequential`, MSE, cross-entropy, SGD, momentum, and Adam. |
| **Presentation layer** | React 19, Vite, canvas/SVG graph views, responsive CSS, deferred advanced-study loading, and a lightweight ambient canvas field. |

## Local study data and parity evidence

The digit study uses a fixed 60-example subset of MNIST, with six samples per digit downsampled from 28 × 28 to 14 × 14 during repository preparation. The generated fixture is committed to the application, so visitors do not download MNIST while using the site.[3]

For an independent numerical check, a fixed two-layer float64 network is evaluated by both the custom engine and PyTorch 2.13.0+cpu using identical inputs, parameters, targets, forward operations, and reverse gradients.

| Parity measurement | Recorded result |
| --- | --- |
| Maximum absolute difference | `5.551115123125783e-17` |
| Largest output difference | `1.3877787807814457e-17` |
| Largest gradient difference | `5.551115123125783e-17` |
| Acceptance threshold | `< 1e-10` |
| Result | Pass |

## Quick start

**Prerequisites:** Node.js 22 and pnpm 10.

```bash
git clone https://github.com/mantisdarling/NeuroForge.git
cd NeuroForge
pnpm install --frozen-lockfile
pnpm dev
```

Open the local address printed by Vite. For a production build and preview:

```bash
pnpm build
pnpm preview
```

## Quality checks

The repository has deterministic local checks and a GitHub Actions workflow for pushes and pull requests to `main`.

```bash
pnpm check            # strict TypeScript
pnpm test             # 10 tests across 4 suites
pnpm security:scan    # credential-marker scan across tracked text files
pnpm security:config  # headers, CSP, and CI action-pin validation
pnpm build            # production Vite build
pnpm delivery:check   # gzip delivery budgets
pnpm audit            # production dependency audit
```

| Verification area | Coverage |
| --- | --- |
| **Gradient correctness** | Finite-difference tests for scalar expressions, matrix operations, softmax/log loss, Conv2D kernels, and max-pool routing. |
| **Learning behavior** | Deterministic checks for sine regression, digit-study loss reduction, CNN-shape learning, and optimizer trace construction. |
| **Type and build safety** | Strict TypeScript compilation and a production Vite build. |
| **Supply-chain and configuration checks** | Locked installation, production dependency audit, tracked-file secret markers, full-SHA CI action pins, and deterministic header/CSP validation. |

## Security and delivery posture

NeuroForge intentionally minimizes its application attack surface: it does not accept executable input, process credentials, persist user data, or expose server-side mutation endpoints. The production configuration adds a restrictive Content Security Policy, frame denial, MIME-sniffing protection, referrer controls, permissions limits, cross-origin protections, and HTTPS upgrading. Fingerprinted build assets are served with immutable caching, while the advanced-study code is deferred until a visitor approaches or opens that section.

These controls reduce specific risks; they do **not** guarantee that any website can never be attacked. Hosting-account security, DNS, provider availability controls, branch protection, alert configuration, dependency updates, and incident response remain operational responsibilities.

## Project structure

```text
client/src/
  engine/                  Custom tensor engine, layers, optimizers, and bounded studies
  components/              Graph, lab, guided route, and visual learning surfaces
  App.tsx                  Application composition and deferred advanced-study boundary
tests/                     Gradient, training, convolution, and advanced-study checks
scripts/                   Fixture preparation, parity export, secret scan, and config checks
.github/workflows/ci.yml   Deterministic pull-request and push verification
vercel.json                Static deployment, cache policy, and browser security headers
```

## Deployment

The project is ready for Vercel static deployment. Import the repository, retain `pnpm build` as the build command, and publish the generated `dist` directory. The included `vercel.json` applies the production browser headers and immutable cache policy for fingerprinted assets.

## References

[1] [Baydin, Pearlmutter, Radul, and Siskind, “Automatic Differentiation in Machine Learning: a Survey,” *Journal of Machine Learning Research*, 2018.](http://www.jmlr.org/papers/v18/17-468.html)

[2] [Auto-eD, “Module 3: The Reverse Mode of Automatic Differentiation.”](https://auto-ed.readthedocs.io/en/latest/mod3.html)

[3] [TensorFlow Datasets, “MNIST.”](https://www.tensorflow.org/datasets/catalog/mnist)
