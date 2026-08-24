# Contributing to NeuroForge

Thank you for improving NeuroForge. Contributions should preserve the project’s purpose: making reverse-mode automatic differentiation observable through small, browser-local, and verifiable learning experiments.

## Before you begin

Use Node.js 22 and pnpm 10. Create a focused branch from the current `main` branch, keep the change set narrow, and explain the learner-facing value in the pull request description.

```bash
pnpm install --frozen-lockfile
pnpm check
pnpm test
pnpm security:scan
pnpm security:config
pnpm build
pnpm delivery:check
pnpm audit
```

## Contribution standards

| Area | Expectation |
| --- | --- |
| **Math and learning behavior** | Add or update deterministic tests for changed tensor operations, gradients, optimizers, or training behavior. Keep numerical claims observable and reproducible. |
| **User experience** | Preserve the Signal Observatory visual system, keyboard access, visible focus states, responsive layouts, high-contrast mode, and reduced-motion support. |
| **Runtime boundaries** | Keep experiments bounded and browser-local. Do not add browser machine-learning runtimes, unbounded runtime data downloads, credentials, or server-side mutation paths without an agreed architecture change. |
| **Security** | Never commit keys, tokens, or personal data. Keep Content Security Policy and release-configuration checks passing. Report suspected vulnerabilities privately to the repository owner rather than opening a public issue with exploit details. |
| **Documentation** | Update `README.md` when behavior, commands, supported studies, security posture, or deployment requirements change. |

## Pull request checklist

Describe what changed, why it improves the project, and how it was validated. Include screenshots only for visible interface changes. Confirm that the quality commands above pass, that no generated build output is committed, and that the change does not introduce unrelated formatting churn.

By submitting a contribution, you agree that it may be distributed under the repository’s [MIT License](LICENSE).
