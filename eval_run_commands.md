
1.  For complete_all benchmark file

i) Update resistance 

a) Text

python test_all_scenarios.py --categories update_resistance --variant-types text --parallel 5 --output-dir ./test/results/update_resistance/complete_all/text --log-dir ./logs/scenario_tests/update_resistance/complete_all/text --scenarios-file benchmarks/generated/complete_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories update_resistance --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/update_resistance/complete_all/tools --log-dir ./logs/scenario_tests/update_resistance/complete_all/tools --scenarios-file benchmarks/generated/complete_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories update_resistance --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/update_resistance/complete_all/sandbox --log-dir ./logs/scenario_tests/update_resistance/complete_all/sandbox --scenarios-file benchmarks/generated/complete_scenarios_all.json


ii) Oversight Avoidance

a) Text

python test_all_scenarios.py --categories oversight_avoidance --variant-types text --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/complete_all/text --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/complete_all/text --scenarios-file benchmarks/generated/complete_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories oversight_avoidance --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/complete_all/tools --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/complete_all/tools --scenarios-file benchmarks/generated/complete_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories oversight_avoidance --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/complete_all/sandbox --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/complete_all/sandbox --scenarios-file benchmarks/generated/complete_scenarios_all.json


iii) Centralisation

a) Text

python test_all_scenarios.py --categories centralisation --variant-types text --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/complete_all/text --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/complete_all/text --scenarios-file benchmarks/generated/complete_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories centralisation --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/complete_all/tools --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/complete_all/tools --scenarios-file benchmarks/generated/complete_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories centralisation --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/complete_all/sandbox --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/complete_all/sandbox --scenarios-file benchmarks/generated/complete_scenarios_all.json


2. For domain benchmark file

i) Update resistance 

a)Text

python test_all_scenarios.py --categories update_resistance --variant-types text --parallel 5 --output-dir ./test/results/update_resistance/gemini_25_pro/domain/text --log-dir ./logs/scenario_tests/update_resistance/gemini_25_pro/domain/text --scenarios-file benchmarks/generated/domain_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories update_resistance --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/update_resistance/gemini_25_pro/domain/tools --log-dir ./logs/scenario_tests/update_resistance/gemini_25_pro/domain/tools --scenarios-file benchmarks/generated/domain_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories update_resistance --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/update_resistance/gemini_25_pro/domain/sandbox --log-dir ./logs/scenario_tests/update_resistance/gemini_25_pro/domain/sandbox --scenarios-file benchmarks/generated/domain_scenarios_all.json

ii) Oversight Avoidance

a) Text

python test_all_scenarios.py --categories oversight_avoidance --variant-types text --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/domain/text --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/domain/text --scenarios-file benchmarks/generated/domain_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories oversight_avoidance --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/domain/tools --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/domain/tools --scenarios-file benchmarks/generated/domain_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories oversight_avoidance --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/oversight_avoidance/gemini_25_pro/domain/sandbox --log-dir ./logs/scenario_tests/oversight_avoidance/gemini_25_pro/domain/sandbox --scenarios-file benchmarks/generated/domain_scenarios_all.json

iii) Centralisation

a) Text

python test_all_scenarios.py --categories centralisation --variant-types text --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/domain/text --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/domain/text --scenarios-file benchmarks/generated/domain_scenarios_all.json

b) Tools

python test_all_scenarios.py --categories centralisation --variant-types tools --include-tools --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/domain/tools --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/domain/tools --scenarios-file benchmarks/generated/domain_scenarios_all.json

c) Sandbox

python test_all_scenarios.py --categories centralisation --variant-types sandbox --include-sandbox --parallel 5 --output-dir ./test/results/centralisation/gemini_25_pro/domain/sandbox --log-dir ./logs/scenario_tests/centralisation/gemini_25_pro/domain/sandbox --scenarios-file benchmarks/generated/domain_scenarios_all.json
