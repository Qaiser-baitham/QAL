# Archived Legacy Files
Archived on: 2026-04-17
Reason: These files are not imported by the active pipeline (main.py → cli.py → runner.py).
Each was replaced by a newer module. Kept here for reference if any formula or logic needs recovery.

| Original Path | Replaced By | Notes |
|---------------|-------------|-------|
| nn_model/models.py | models/factory.py | Old MLP/CNN definitions with QuantLinear integration |
| nn_model/quant_layers.py | hardware_sim/effects.py | STE quantized layers; effects.py does this via perturbed_weights |
| data_loader/dataset_loader.py | data_loader/datasets.py | Duplicate loader with slightly different normalization stats |
| data_loader/memristor_loader.py | data_loader/raw_memristor.py | Older parser without auto-column detection |
| device_model/characterization.py | device_model/extractor.py | Old extraction logic depending on memristor_loader |
| device_model/device.py | hardware_sim/effects.py | MemristorDevice behavioral sim; effects.py reimplements inline |
| training/trainer.py | training/runner.py | Standalone training loop, superseded by ExperimentRunner |
| utils/config.py | cli.py JSON loading | YAML-based Config class, never used |
| utils/traces.py | (never integrated) | TraceRecorder for layer activations; hooks never registered |
| visualization/plots.py | visualization/exporter.py | Older plotting functions, fully replaced |
| configs/default.yaml | configs/default.json | YAML config template, CLI uses JSON |
