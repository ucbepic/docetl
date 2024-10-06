export YML_KEY=pacc

if ! docetl build example_data/pacc/$YML_KEY.yaml; then
    echo "docetl build failed"
    exit 1
fi

if ! docetl run example_data/pacc/${YML_KEY}_opt.yaml; then
    echo "docetl run failed"
    exit 1
fi

if ! python3 example_data/pacc/json2html.py example_data/pacc/${YML_KEY}_analysis.json ; then
    echo "python3 script failed"
    exit 1
fi



