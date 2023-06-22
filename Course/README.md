## About

This folder contains sample code for the course “GPU Programming Primitives for Computer Graphics”. 


## Build and Run

One would require [premake](https://premake.github.io/) and [Visual Studio](https://visualstudio.microsoft.com/) to build and run on Windows.

The following instruction is an example of how to generate Visual Studio 2022 solution.

`../tools/premake5/win/premake5.exe vs2022`

This instruction also copies the dll files needed to run HIP programs. 

These examples utilize [Orochi](https://github.com/GPUOpen-LibrariesAndSDKs/Orochi), a library that can load HIP and CUDA runtimes dynamically. It includes the runtime compiler for HIP; however, CUDA SDK is required to install on its environment. Please install the SDK based on the vender-provided instructions.

