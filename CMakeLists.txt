cmake_minimum_required(VERSION 3.16)
project(AdvancedTimeSeriesForecasting VERSION 25.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

# Source files
set(SOURCES
    src/time_series_forecaster.cpp
    src/quantum_inspired_optimizer.cpp
    src/federated_learning_engine.cpp
)

# Include directories
include_directories(include)

# Executable
add_executable(forecaster ${SOURCES})

if(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(forecaster PRIVATE Threads::Threads)
endif()

if(WIN32)
    target_compile_options(forecaster PRIVATE /W4 /WX)
else()
    target_compile_options(forecaster PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS forecaster DESTINATION bin)
