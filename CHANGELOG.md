# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-m0] - 2026-11-21

### Added

- Repository conventions, license, PyTorch attribution, and the initial
  architecture decision log (M0 "Anvil").
- CMake + Ninja build and Windows/Linux CI, including a `glslc` to SPIR-V
  helper for the Vulkan shaders that arrive in M2.
- C unit-test harness and the dataclass-configured parity harness skeleton.
- Initial `add` operator on the `cpu_ref` backend with PyTorch broadcasting
  semantics; 13/13 parity cases pass bit-exactly against PyTorch CPU and CUDA.
- M0 status report (`docs/m0_status.md`) and committed parity report.

## [Unreleased]
