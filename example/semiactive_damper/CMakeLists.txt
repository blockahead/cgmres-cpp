cmake_minimum_required(VERSION 3.0.0)
get_filename_component(dirname "${CMAKE_CURRENT_SOURCE_DIR}" NAME)
project("${dirname}")

# Compile options
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options(-O3 -Wall)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
    add_compile_options(-O3 -Wall)
elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    add_compile_definitions(PRIVATE _USE_MATH_DEFINES)
endif()

# Executables
add_executable(${PROJECT_NAME}
    main.cpp
    # ${CGMRES_SOURCE_DIR}/src/cgmres.cpp
    ${CGMRES_SOURCE_DIR}/src/gmres.cpp
)
target_include_directories(${PROJECT_NAME}
    PUBLIC ${CGMRES_SOURCE_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
    openblas
)
