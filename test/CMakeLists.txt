cmake_minimum_required(VERSION 3.0.0)

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options(-O3 -Wall)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    add_compile_options(-O3 -Wall)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    add_compile_definitions(PRIVATE _USE_MATH_DEFINES)
endif()

# Enable Google test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# Test for cblas
add_executable(cblas_test
    cblas_test.cpp
)
target_include_directories(cblas_test
    PRIVATE ${CGMRES_SOURCE_DIR}/include
)
target_link_libraries(cblas_test
    openblas
    GTest::gtest_main
)
gtest_discover_tests(cblas_test)
