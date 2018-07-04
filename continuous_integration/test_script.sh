python --version

run_tests() {
    TEST_CMD="nosetests -s -v"

    
    mkdir -p $TEST_DIR
    cd $TEST_DIR
    TEST_CMD="$TEST_CMD --with-coverage pastis --cover-package pastis"
    $TEST_CMD pastis
}


if [[ "$SKIP_TESTS" != "true" ]]; then
    run_tests
fi 
