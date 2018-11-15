#!/bin/bash
rm crpm/__pycache__/*.pyc
pylint crpm
pytest
