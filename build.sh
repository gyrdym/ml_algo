#!/bin/bash

pub run build_runner build
find . -size 0 -delete
./clear_di_cache.sh
