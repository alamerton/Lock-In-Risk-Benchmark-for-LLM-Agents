NADN AISI Inspect Docker Plugin Sandbox
Structure: Split by federated node (CAA, NATS, Southern, Northern, Scotland)
Data size: Full stress dataset (C) - several thousand rows across CSVs

Folders and key files:
- /CAA/
    - certified_operators.csv
    - approved_vehicle_types.csv
    - operator_fleet.csv
- /NATS/
    - permanent_restricted_airspace.csv
    - national_corridors.csv
- /Southern/, /Northern/, /Scotland/
    - regional_flight_plans.csv
    - live_telemetry.csv
    - local_temporary_restrictions.csv
    - local_weather_and_terrain.csv
- /Agent_Directive/
    - phase1.txt
    - phase2.txt
- /prompts/
    - phase1_prompt.txt
    - phase2_prompt.txt

Use:
- These CSVs are synthetic but structured to reflect the schema described in the directive.
- The Agent_Directive prompts are ready to be used by an agentic model in the AISI Inspect Docker Plugin environment.
- A zip archive of the full package is available at '/mnt/data/nadn_aisi_sandbox.zip'.
