# Probabilistic Data Association

<!-- TABLE OF CONTENTS -->
## Table of content
* [Introduction](#Introduction)
* [Description](#Description)
* [Project installation and requirements](#Project-installation-and-requirements)
* [Using the filter](#Using-the-filter)
* [Additional tools](#Additional-tools)
* [Project Structure](#Project-Structure)
* [Final notes](#Final-notes)

## Introduction
This project is part of the final project in Estimation theory given by Prof. Yaakov Oshman 
in the Technion institute of technology.
The work was done by Shahar Avitan and Ori Einy.

## Description
This project implements the Probabilistic Data Association Filter (PDAF)
and uses simulation to demonstrate and examine the filter's performance.

## Project installation and requirements
The project is based on **python3.10**
In the requirements.txt file all the other packages used can be found.

To use the project, pull it to your local machine, and install
the necessary packages from the requirements.txt file.

## Using the filter
The filter is designed to handle many cases and therefore must receive all 
information about the system being used. From the dynamic matrix to the 
noise of the system.
It can be seen clearly in the different example we ran.

To run each simulation you can either call main.py and give the
desired simulation name or call the simulation file itself.

## Additional tools
As part of the project we have created additional tools.
* Plotter: The plotter has the ability to plot several types of data on the same plot
and choose between scatter plot or line plots. 
It has a capability of simulation the specific problems we have created in the basic simulation.
* DataGeneration: In this folder we combined all the generated data classes, from the target we intend to track
to the different clutter that should be measured by our filter.

## Project Structure
```
ProbabilisticDataAssociationFilter
│   .gitignore
│   main.py
│   README.md
│   requirements.txt
│   test_filter.py
│
├───DataGeneration
│   │   Clutter.py
│   │   Satellite.py
│   │   SpaceClutter.py
│   │   Target.py
│   │   __init__.py
│
├───ProbabilisticDataAssociation
│   │   ProbabilisticDataAssociationFilter.py
│   │   __init__.py
│   │
│
├───Simulations
│   │   basic_simulation.py
│   │   distribution_simulation.py
│   │   location_initial_condition_simulation.py
│   │   measurement_noise_simulation.py
│   │   monte_carlo_simulation.py
│   │   parameter_simulation.py
│   │   Pd_simulation.py
│   │   Pg_simulation.py
│   │   satellite_simulation.py
│   │   velocity_initial_condition_simulation.py
│   │   __init__.py
│
├───Tools
│   │   earth_from_space.png
│   │   Plotter.py
│   │   __init__.py
```

## Final notes
The project was derived mathematical under certain assumptions
in accordance to Yaakov Bar Shalom's paper on PDA filter's.
It is highly advised to read and understand the underlying assumption 
of the filter before using it for specific implementations.