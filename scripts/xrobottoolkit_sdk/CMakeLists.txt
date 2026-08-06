cmake_minimum_required(VERSION 3.14)
project(PXREARobotSDKSourceBuild LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROTOBUF REQUIRED protobuf)
pkg_check_modules(GRPCPP REQUIRED grpc++)

add_library(PXREARobotSDK SHARED
    "${PXREA_SDK_SRC_DIR}/PXREARobotSDK.cpp"
    "${PXREA_PROTO_DIR}/PXREAService.pb.cc"
    "${PXREA_PROTO_DIR}/PXREAService.grpc.pb.cc"
)

target_compile_definitions(PXREARobotSDK PRIVATE
    PXREACLIENTSDK_LIBRARY
    PXREAROBOTSDK_LIBRARY
    "${PXREA_LINUX_DEFINE}"
)

target_include_directories(PXREARobotSDK PRIVATE
    "${PXREA_SDK_SRC_DIR}"
    "${PXREA_PROTO_DIR}"
    ${PROTOBUF_INCLUDE_DIRS}
    ${GRPCPP_INCLUDE_DIRS}
)

target_compile_options(PXREARobotSDK PRIVATE
    ${PROTOBUF_CFLAGS_OTHER}
    ${GRPCPP_CFLAGS_OTHER}
    -include unistd.h
)

target_link_directories(PXREARobotSDK PRIVATE
    ${PROTOBUF_LIBRARY_DIRS}
    ${GRPCPP_LIBRARY_DIRS}
)

target_link_libraries(PXREARobotSDK PRIVATE
    ${GRPCPP_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    Threads::Threads
    dl
)

set_target_properties(PXREARobotSDK PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PXREA_OUTPUT_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
