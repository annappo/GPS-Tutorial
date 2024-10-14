# Tutorial: Real-Time GPS Tracking using RTL-SDR and Python

## Introduction
---

This tutorial has two objectives: 

- On the one hand, it is intended as an introduction to satellite navigation using the *Global Positioning System* (GPS). The programming of a *Software Defined Radio* (SDR) is used to demonstrate how the radio signals sent by the GPS satellites can be received and processed. For this purpose, the text is structured in the style of an *interactive tutorial* (JupyterLab notebook) with short explanations and code examples.
  
- On the other hand, the text shall serve as documentation of a Python program for real-time GPS tracking. The code is developed and commented on the basis of many examples and then integrated as Python classes into the scripts. The final program for real-time positioning and tracking has been published separately on Github: [https://github.com/annappo/GPS-SDR-Receiver](#https://github.com/annappo/GPS-SDR-Receiver).  

The tutorial is aimed in particular at people who

- want to acquire theoretical and practical knowledge of satellite navigation,
- prefer *learning by doing* to better understand abstract concepts,
- would like to carry out GPS tracking in real-time based on an SDR,
- enjoy programming of devices.

If you are not familiar with *JupyterLab*, you can start with the PDF version of the tutorial. Beginners are offered help with the installation of *Python* and *JupyterLab* in the appendix.


## Installation
---

The files and folders of the project are organized in the following way (dots indicate a list of files):

    .
    ├── README.md
    ├── LICENSE
    ├── requirements.txt
    ├── GPS-Tutorial-en.ipynb
    ├── data
    │   └── ...
    ├── figures
    │   └── ...
    ├── pycode
        └── ...

Simply copy all files to a directory of your choice by retaining the given folder structure. This can be done by downloading the project as ZIP file or by cloning the repository, 

```
git clone https://github.com/annappo/GPS-Tutorial
```

Besides a JupyterLab installation, the following modules are required for running the code examples:
 
    matplotlib
    numpy
    scipy
    pyrtlsdr[lib]
    setuptools

If these modules are not yet available on your system, it is recommended to use either an Anaconda distribution or to create a virtual environment for installation. Depending on the operating system or Python distribution used, you may encounter problems when installing the RTL-SDR driver. You can find help on this and further information for the software installation in the appendix of the tutorial.

## Abstract
---

The instrumental basis of the tutorial is a low-budget USB stick that was originally developed for the reception of radio programs and television via DVB-T (Digital Video Broadcasting - Terrestrial). Its main function is to receive a radio signal at an adjustable frequency, digitize it and then transfer the data to a computer via the USB interface. All subsequent steps, in particular the decoding of the information contained in the data, are performed by a computer instead of using analog electronic components of a conventional radio. This approach is known as *Software Defined Radio* (SDR). 

The focus of the tutorial is on the step-by-step development of efficient Python code for measuring and decoding GPS data. The mathematical treatment is limited to the bare essentials to keep the text simple and make it easy for beginners to get started. Nevertheless, it is sometimes unavoidable to use advanced methods such as complex numbers and Fourier transforms. An introduction to these methods can be found in the appendix. For a more comprehensive treatment of GPS navigation, references are given at the end of the tutorial.


### Table of Contents (short version)

* [GPS navigation](#GPS-navigation)
    * [Basics](#Basics)
    * [Data reception](#Data-reception)
    * [Data decoding](#Data-decoding)
    * [Positioning](#Positioning)
* [GPS tracking using an RTL-SDR](#GPS-tracking-using-an-RTL-SDR)
    * [First steps](#First-steps)
    * [Noise analysis](#Noise-analysis)
    * [Observation of single satellites](#Observation-of-single-satellites)
    * [Observation of many satellites](#Observation-of-many-satellites)
* [Real-time tracking](#Real-time-tracking)
    * [Code modifications](#Code-modifications)
    * [Graphical User Interface](#Graphical-User-Interface)
    * [Installation](#Installation)
* [Appendix](#Appendix)
    * [RTL-SDR: Features and accessories](#RTL-SDR:-Features-and-accessories)
    * [Python and JupyterLab](#Python-and-JupyterLab)
    * [Computer performance](#Computer-performance)
    * [Mathematical additions](#Mathematical-additions)
* [Notes](#Notes)
* [References](#References)

