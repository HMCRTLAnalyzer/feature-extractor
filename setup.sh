#!/bin/bash

echo Building aiger library
cd aiger && ./configure.sh && make
cd ..; ln -s aiger/aigtodot aigtodot
pip install xgboost pandas matplotlib numpy sklearn
