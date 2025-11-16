add_executable(ngp-minimal-app
        main_minimal.cu
)

target_link_libraries(ngp-minimal-app PRIVATE ngp-baseline-nv)

