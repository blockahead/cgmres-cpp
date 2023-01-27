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

add_executable(matrix_test
    matrix_test.cpp
)
target_include_directories(matrix_test
    PRIVATE ${CGMRES_SOURCE_DIR}/include
)
target_link_libraries(matrix_test
    GTest::gtest_main
)
gtest_discover_tests(matrix_test)
