cmake_minimum_required(VERSION 3.0.0)

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options(-O3 -Wall)
endif()

# C/GMRES root directory
# get_filename_component(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

# Executables
add_executable(${PROJECT_NAME}
    main.cpp
    # ${CGMRES_SOURCE_DIR}/src/cgmres.cpp
    ${CGMRES_SOURCE_DIR}/src/gmres.cpp
)
target_include_directories(${PROJECT_NAME}
    PUBLIC ${CGMRES_SOURCE_DIR}/include
)

# enable_testing()
