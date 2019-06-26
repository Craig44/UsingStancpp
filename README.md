# UsingStancpp
A little stand alone repo for testing how to use Stan in a C++ standalone exe

## What is this
I was playing around with how to use Stan's math library and MCMC functionality, as I really want to port it into Casal2 (https://github.com/NIWAFisheriesModelling/CASAL2). Before I was confident in doing that
I wanted a proof of concept which is what this repo is about. There are basically three models defined in "8Schools.hpp" "linear_regression.hpp" and "Rosenbrock.hpp" (although I don't think I use this). 
These models classes are called in SimpleModel.cpp where I play around with them and learn how to use them. The 8Schools example was my first learning point I asked rstan to print the C++ files for that model 
and then tried to run it myself using stans optimisers and mcmc algorithms. Next I wanted to see what methods and members of a stan Model class were needed so I made a simple linear regression model
example and you can see how that all works SimpleModel.cpp.

## Building SimpleMpdel.exe
In order to compile SimpleMpdel.exe you will need to download or clone the stan dependencies. This information lies in the file 'BuildCommand.txt'

