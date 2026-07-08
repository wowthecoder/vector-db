add_executable(vectordb_tests
    vector_db_tests.cpp
)

target_link_libraries(vectordb_tests
    PRIVATE vectordb
)

target_compile_options(vectordb_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

add_test(NAME vectordb_tests COMMAND vectordb_tests)
