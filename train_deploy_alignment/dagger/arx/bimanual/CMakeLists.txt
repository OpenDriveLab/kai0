cmake_minimum_required(VERSION 3.10)
project(arx_x5_python VERSION 1.0)

find_package(pybind11 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/arx_x5_src/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/arx_hardware_interface/include)

pybind11_add_module(${PROJECT_NAME} src/single_arm_interface.cpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Target architecture is ARM")
    set(DEPEND_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_x5_src/libarx_x5_src-arm64.so")
else ()
    set(DEPEND_LIB "${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_x5_src/libarx_x5_src.so")
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC ${DEPEND_LIB} pybind11::module)

# 设置安装路径
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/api/" CACHE PATH "Installation Directory" FORCE)

install(DIRECTORY lib/ DESTINATION ./)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_x5_src/libarx_x5_src-arm64.so
    	    DESTINATION ./arx_x5_src
    	    RENAME libarx_x5_src.so)
endif()
install(TARGETS ${PROJECT_NAME}
        LIBRARY DESTINATION ./${PROJECT_NAME}/
        ARCHIVE DESTINATION ./${PROJECT_NAME}/)
