# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `FullyConnected` no longer panics (`assertion failed: idx < self.len()`) when the model omits the optional bias tensor, as the TFLite converter does for all-zero biases (e.g. `use_bias=False`)

## [0.1.3] - 2024-06-01

### Fixed

- Update to new `esp-hal` version

## [0.1.2] - 2024-02-09

### Added

- MicroFlow is now available on [crates.io](https://crates.io/crates/microflow) 🎉

### Fixed

- Fix Arduino Uno example
- Update to new `esp-hal` version

### Changed

- Minor documentation changes

## [0.1.0] - 2023-07-02

Initial release.
