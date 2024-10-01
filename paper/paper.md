---
title: "SONIC: Sound Organization and Network Integration for Collection and Collaboration"
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Daniel Kadyrov
    orcid: 0000-0002-5390-6346
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Dr. Alexander Sutin
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Dr. Alexander Sedunov
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: 2
affiliations:
  - name: Stevens Institute of Technology, USA
    index: 1
    ror: 00hx57361
  - name: IEEE, USA
    index: 2
  - name: University of Houston, USA
    index: 3
date: 13 August 2017
bibliography: reference.bib
# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The collection, organization, and sharing of acoustic data is a vital component for analysis, research, collaboration, and to support the advancement of machine learning and artificial intelligence capabilities that depend heavily on large amounts of data. The standard method of organizing and publishing acoustic data consists of audio files with supplemental documentation describing the recording metadata. This method becomes complex when the audio data is unsegmented and additional materials are elaborate and complicated. The Sound Organization and Network Integration for Collection and Collaboration (SONIC) package was developed to accommodate acoustic research by leveraging relational database platforms and industry standard Python packages to facilitate acquisition, structuring, labeling, extraction, and sampling. This package is designed to be flexible, parametric, extensible, and integrable into a variety of possible acoustic applications. SONIC utilizes `SQLalchemy`, a Python SQL toolkit and Object-Relational Mapping (ORM) library, to interact with a database describing the audio filepaths and the associated metadata information on the recording subjects, events, sensors and channels, generated samples, and classification results. The database is currently implemented using SQLite, a serverless single-file engine that can be shared agnostic of operating system or programming languages and can easily be migrated and scaled to other database engines such as MySQL, PostgreSQL, or Oracle. Furthermore, server packages such as `Flask`, `FastAPI`, or `Django` can be used to facilitate network integration for data acquisition from sensors, visualization, and sharing. Although SONIC is written in Python, the database can be accessed using any programming language that supports SQL including R, MATLAB, and Java.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`Gala` was designed to be used by both astronomical researchers and by
students in courses on gravitational dynamics or astronomy. It has already been
used in a number of scientific publications [@Pearson:2017] and has also been
used in graduate courses on Galactic dynamics to, e.g., provide interactive
visualizations of textbook material [@Binney:2008]. The combination of speed,
design, and support for Astropy functionality in `Gala` will enable exciting
scientific explorations of forthcoming data releases from the _Gaia_ mission
[@gaia] by students and experts alike.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001` -> "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
