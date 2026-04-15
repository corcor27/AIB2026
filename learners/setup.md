---
title: Setup
---

## Overview

This lesson is designed to be run on a personal computer.
All of the software and data used in this lesson are freely available online,
and instructions on how to obtain them are provided below.

## Install Python

In this lesson, we will be using Python 3 with some of its most popular scientific libraries.
One can install a plain-vanilla Python and all required libraries by hand. If you've followed
the [Carpentry workshop installation instructions](https://carpentries.github.io/workshop-template/install_instructions/#python-1)
you will have installed miniforge and the required libraries in an environment called `carpentries`. 
Before starting you'll want to type `conda activate carpentries` in order to activate that environment.

## Obtain lesson materials

1. Download [python-novice-inflammation-data.zip][zipfile1], [python-novice-inflammation-code.zip][zipfile2], [iris.csv][iris] and [additional_stuff.zip][Additional].
2. Create a folder called `swc-python` on your Desktop.
3. Move downloaded files to `swc-python`.
4. Unzip the files.

You should see three folders called `data`, `code`, and additional stuff in the `swc-python` directory on your
Desktop. You should also see the iris dataset in that folder.

## Launch Python interface

To start working with Python, we need to launch a program that will interpret and execute our
Python commands. Below we list several options. If you don't have a preference, proceed with the
top option in the list that is available on your machine. Otherwise, you may use any interface
you like.

## Option A: Jupyter Notebook

A Jupyter Notebook provides a browser-based interface for working with Python.
You can launch a notebook from the command line:

:::::::::::::::: spoiler

## Command line (Terminal)

1\. Navigate to the `data` directory:

::::::::::::::::: spoiler

## Unix shell

If you're using a Unix shell application, such as Terminal app in macOS, Console or Terminal
in Linux, or [Git Bash][gitbash] on Windows, execute the following command:

```bash
cd ~/Desktop/swc-python/data
```

:::::::::::::::::::::::::

::::::::::::::::: spoiler

## Command Prompt (Windows)

On Windows, you can use its native Command Prompt program.  The easiest way to start it up is
pressing <kbd>Windows Logo Key</kbd>\+<kbd>R</kbd>, entering `cmd`, and hitting
<kbd>Return</kbd>. In the Command Prompt, use the following command to navigate to
the `data` folder:

```source
cd /D %userprofile%\Desktop\swc-python\data
```

:::::::::::::::::::::::::

2\. Start Jupyter server

::::::::::::::::: spoiler

## Unix shell

```bash
jupyter notebook
```

:::::::::::::::::::::::::


::::::::::::::::: spoiler

## Command Prompt (Windows)

```source
python -m notebook
```

:::::::::::::::::::::::::

3\. Launch the notebook by clicking on the "New" button on the right and selecting "Python 3"
from the drop-down menu:
![](fig/jupyter-notebook-launch-notebook2.png){alt='Anaconda Navigator Notebook directory'}

:::::::::::::::::::::::::

  <!-- vertical spacer -->

## Option B: IPython interpreter

IPython is an alternative solution situated somewhere in between the plain-vanilla Python
interpreter and Jupyter Notebook. It provides an interactive command-line based interpreter with
various convenience features and commands.  You can install IPython on your system
[system](https://ipython.org/install.html).

To start using IPython, execute:

```source
ipython
```

  <!-- vertical spacer -->

## Option C: plain-vanilla Python interpreter

To launch a plain-vanilla Python interpreter, execute:

```source
python
```

If you are using [Git Bash on Windows][gitbash], you have to call Python *via* `winpty`:

```source
winpty python
```


[zipfile1]: data/python-novice-inflammation-data.zip
[zipfile2]: ../episodes/files/code/python-novice-inflammation-code.zip
[gitbash]: https://gitforwindows.org
[iris]: data/iris.csv
[additional]: data/additional_stuff.zip

## Your anaconda environment

### What is an python environment

A Python environment (often called a virtual environment) is an isolated workspace where you can install and manage Python packages independently from other projects.

Think of it like this:

Imagine each project you work on has its own “bubble”:

* It has its own Python version
* Its own libraries (packages)
* Its own dependencies

This prevents conflicts between projects.

### Why Python environments are important

Without environments:

* One project might need pandas version 1.5
* Another might need pandas version 2.0
* These would clash and break things

With environments:

* Each project gets exactly what it needs

### Activate a Conda environment

As part of the installation, you should already have installed Miniforge.
On Windows, open the **Miniforge Prompt**. On macOS or Linux, open a terminal.

You only need to create the environment once, but you need to activate it each time you start a new session.

To activate the `carpentries` environment, type:

```source
conda activate carpentries
```

To check that the environment has activated correctly, you should either see that your prompt now displays the environment name or you can type:

```source
conda info --envs
```

The active environment will have a `*` next to it.

### Install packages with Conda

Most of the packages you need should already be installed in the `carpentries` environment.
Here, we will see how to install packages using Conda in Miniforge or the terminal. Later, we will also see examples of how to install packages from within Jupyter Notebook.

```source
conda install numpy
```
Another option if conda is problematic is to use, then you can install packages via pip

```source
pip install numpy
```

Make sure that, when you run these commands, you are in the correct environment (carpentries). 

:::::::::::::::::::::::::::::::::::::::::  callout
## Safe guarding "base"

In Anaconda, the default environment is called “base.” It serves as the core foundation for everything within Anaconda. If this environment becomes corrupted or broken, you may lose the ability to create or install new environments. For this reason, it’s recommended that you only use the base environment to create new environments or update Conda, and avoid making other changes to it.

::::::::::::::::::::::::::::::::::::::::::::::::::
## Jupyter Notebook
### What is Jupyter Notebook?

Jupyter Notebook is an interactive tool that lets you write and run code in small sections, while also adding notes, explanations, and visual outputs, all in one place.

It’s widely used in:

* Data science
* Machine learning
* Research and education

Think of it as a combination of a coding environment and a document.

Understanding Cells

A notebook is made up of cells, which are the building blocks of your work.

Types of cells:
* Code cells → where you write and run Python code
* Markdown cells → where you write text, explanations, or instructions

### Using Jupyter Notebook

![](fig/creating_file.png){alt='Click the red box to create a new notebook and then select python 3 (ipykernal)'}

Click the red box to create a new notebook and then select python 3 (ipykernel)

![](fig/notebook.png){alt='Blue: the cells where you run code, Orange: is the output from a cell above, Pink: is the current cell selected, Red: remove selected cell, Green: Run select cell and Black: Name of worksheet, change by clicking it.'}

Blue: the cells where you run code, Orange: is the output from a cell above, Pink: is the current cell selected, Red: remove selected cell, Green: Run select cell and Black: Name of worksheet, change by clicking it.

![](fig/running_cells.png){alt='Red: run selected cell, Blue: run all if needing to check of errors and Green is to start and run cells with a blank canvas.'}

Red: run selected cell, Blue: run all if needing to check of errors and Green is to start and run cells with a blank canvas.

### Why Use Notebooks Instead of Terminal or IDE?

Jupyter Notebooks are used as a starting point because they are:

* Beginner-friendly
* No need to manage multiple files
* Immediate feedback when running code
* Interactive, run small pieces of code step-by-step
* Experiment or test small sections of code
* Visual Outputs, charts, and explanations appear together
* Great for learning
* You can mix code + commenting + results in one place


### A Quick Note on Terminal & IDEs

More advanced workflows often use:

* The command line (terminal)
* Full development environments (IDEs)
* High-performance computing (HPC) systems for large-scale workloads

These are powerful, but:

* Require more setup
* Are less interactive for beginners

### Why We Start with Notebooks

Notebooks provide the most accessible entry point because they:

* Reduce complexity
* Encourage experimentation
* Help you understand code step-by-step

Once you’re comfortable, you can then move on to:

* Scripts
* Terminals
* Larger systems like HPC


