#!/bin/bash

echo Building aiger library
cd aiger && ./configure.sh && make
ln -s aiger/aigtodot aigtodot
