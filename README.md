# UHPC Pull-Out Predictor for Windows

This repository provides a portable graphical user interface for predicting the pull-out response of steel fibers embedded in ultra-high-performance concrete (UHPC). The application implements a physics-informed two-stage framework that predicts:

- peak pull-out load (`Pmax`);
- slip corresponding to the peak load;
- the complete pull-out load-slip curve from 0 to 5 mm (251 points).

## Download and run

1. Download `UHPC_Pullout_Predictor_Windows.zip`.
2. Extract the complete ZIP archive.
3. Double-click `UHPC_Pullout_Predictor.exe` in the extracted folder.

The portable package includes the trained model and Python runtime. Users do not need to install Python, machine-learning libraries, or the original training database. The package is intended for 64-bit Windows 10 and Windows 11.

Do not move the executable out of the extracted folder because it uses the accompanying `runtime` and `artifacts` directories.

Windows SmartScreen may display an unrecognized-publisher warning because the research executable is not digitally signed.

## Inputs

The GUI accepts 12 variables describing fiber geometry, fiber mechanical properties, specimen configuration, matrix properties, curing conditions, and loading conditions:

- fiber shape;
- embedded length;
- inclination angle;
- tensile strength;
- fiber length;
- fiber diameter;
- aspect ratio;
- pull-out specimen shape;
- loading rate;
- water-to-binder ratio;
- compressive strength;
- curing age.

The interface also provides input validation, applicability-domain warnings, automatic aspect-ratio calculation, CSV export, and high-resolution figure export.

## Source code

The `src` directory contains the GUI and prediction source code. The distributed model uses a physics-scaled peak-response ensemble followed by Route-D PCA curve reconstruction.

## Citation

If this software is used in academic work, please cite the associated article describing the physics-informed two-stage framework for small-data prediction of steel-fiber pull-out load-slip responses of UHPC.

## License and data

The executable is provided for research use. The trained model is embedded in the portable package; the original literature-derived training database is not distributed in this repository.
