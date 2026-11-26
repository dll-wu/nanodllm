# Security Policy

## Supported versions

This repository is an early research preview. Security fixes are made only on
the current development branch; older snapshots and generated artifacts are
not supported.

## Reporting a vulnerability

Use the private vulnerability-reporting feature on the repository's hosting
platform once the public preview is published. Include a minimal reproduction,
the affected revision, the expected impact, and any suggested mitigation.

Do not report suspected vulnerabilities in a public issue, discussion, pull
request, or social-media post. If the hosting platform does not offer a private
reporting channel, use a private maintainer contact advertised on that platform
rather than disclosing the issue publicly.

Please avoid accessing other people's data, running disruptive tests, or
retaining sensitive information while investigating a report.

## Research software notice

nano-dllm is experimental training software, not a hardened hosted service.
Users are responsible for securing their datasets, credentials, checkpoints,
logs, compute environments, and any models or services they build with it.

## Model files

Use the weights-only loader for exchanged model files and verify the expected digest from the model publisher. Native training checkpoints and exact-resume files use Python pickle and require trust in the producer. The [portable weights guide](docs/portable-weights.md) gives the export procedure.
