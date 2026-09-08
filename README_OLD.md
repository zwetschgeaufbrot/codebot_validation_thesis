# (Automated) Assessment of the Narrative Description of Analysis Code – A CodeBot Validation Study 

## Overview
This README organises the online materials corresponding to the bachelor's thesis of Anna T. Wittmann.
To reproduce the analyses, download all online materials to your local machine, keeping the files and folders structured as outlined below.
Next, open the Rproj in RStudio.
For the a priori and posteriori precision analyses, run full_sim.qmd and bachelor_sim.qmd.
All plots created in R can be reproduced with the help of full_sim_visual.qmd.
To recreate both the pilot and main analyses run processing.qmd followed by analysis.qmd.
In contrast, the adjudicator script is included because it was saved to the OSF together with my pre-registration and CodeBot and my recorded data previous to the analysis process.

## Structure
```
├── code
│   ├── main
│   │   ├── analysis.qmd                             # detailed analysis process
│   │   └── processing.qmd                        # detailed processing protocol
│   ├── pilot
│   │   ├── analysis.qmd                             # detailed analysis process
│   │   └── processing.qmd                        # detailed processing protocol                         
│   └── precision_analysis
│       ├── bachelor_sim.qmd         # sim of n = 17 a priori precision analysis
│       ├── full_sim.qmd               # sim of full a priori precision analysis
│       └── full_sim_visual.qmd          # visual analysis of precision analysis
├── codebot-for-psychology.Rproj            # Rproj for reproduction of analyses
├── data
│   ├── outputs
│   │   └── plots                              # plots from full_sim_visual.qmd
│   │       └── precision.png
│   ├── processed                                   # output from processing.qmd
│   │    ├── main
│   │    │   ├── codebook_res_main.xlsx       # codebook for processed main data
│   │    │   └── res_main.rds                             # processed main data
│   │    ├── pilot
│   │    │   ├── codebook_res_pilot.xlsx  # codebook for processed piloting data
│   │    │   └── res_pilot.rds                        # processed piloting data
│   │    └── precision_analysis 
│   │        ├── codebook_power_ba_ci.xlsx                    # codebook to .rds
│   │        ├── codebook_ci_estimates.xlsx                   # codebook to .rds
│   │        ├── power_ba_ci.rds              # CIs of n = 17 a posteriori analysis
│   │        └── simulation_ci_estimates.RDS  # CIs of full_simulation_unnested.rds
│   └── raw
│       ├── main
│       │   ├── adjudicator
│       │   │   ├── codebook_combined_reports.xlsx  # codebook for all combined reports
│       │   │   └── combined_reports               # combined adjudicator output
│       │   │       ├── akan_combined_report.csv
│       │   │       ├── avilles_combined_report.csv
│       │   │       ├── berger_combined_report.csv
│       │   │       ├── besken_combined_report.csv
│       │   │       ├── bloy_combined_report.csv
│       │   │       ├── carvalho_combined_report.csv
│       │   │       ├── cha_combined_report.csv
│       │   │       ├── evans_combined_report.csv
│       │   │       ├── helm_combined_report.csv
│       │   │       ├── hsieh_combined_report.csv
│       │   │       ├── hussey_combined_report.csv
│       │   │       ├── mancassola_combined_report.csv
│       │   │       ├── muenster_combined_report.csv
│       │   │       ├── nguyen_combined_report.csv
│       │   │       ├── radtke_combined_report.csv
│       │   │       ├── yang_combined_report.csv
│       │   │       └── yeung_combined_report.csv
│       │   ├── codebot
│       │   │   ├── link.xlsx                           # CB material input list
│       │   │   └── output                               # CB output full sample
│       │   │       ├── all_papers.csv
│       │   │       ├── all_papers.json
│       │   │       └── codebook_all_papers.xlsx  # codebook for combined CB output 
│       │   └── human
│       │       └── sample_coding_human.xlsx  # manual coding of full sample,contains definition of variable in tab "vars"
│       ├── pilot
│       │   ├── codebot
│       │   │   └── output                 # separate CB output for pilot sample
│       │   │       ├── anvari_2023_codebot_rows.csv
│       │   │       ├── anvari_2023_combined_report.csv
│       │   │       ├── anvari_2023_human_with_matches.csv
│       │   │       ├── anvari_2023_left_join.csv
│       │   │       ├── anvari_2023_mapping.csv
│       │   │       ├── codebot_rows.csv
│       │   │       ├── elson_2020_codebot_rows.csv
│       │   │       ├── elson_2020_combined_report.csv
│       │   │       ├── elson_2020_human_with_matches.csv
│       │   │       ├── elson_2020_left_join.csv
│       │   │       ├── elson_2020_mapping.csv
│       │   │       ├── human_with_matches.csv
│       │   │       └── mapping.csv
│       │   ├── combined_reports            # combined CB output for pilot paper
│       │   │   ├── anvari_2023_combined_report.csv
│       │   │   └── elson_2020_combined_report.csv        
│       │   └── human
│       │           └── pilot_paper.xlsx         # manual coding of pilot sample
│       └── precision_analysis
│           ├── codebook_full_simulation_unnested.xlsx        # codebook to .rds
│           ├── codebook_power_ba.xlsx                        # codebook to .rds
│           ├── full_simulation_unnested.rds     # full sample precision analysis sim
│           └── power_ba.rds                     # sim data for a posteriori analysis
├── methods
│   ├── adjudicator.py                      # preregistered adjudicator workflow
│   ├── codebot                             # CB files kindly provided by J. Cummins
│   │   └── ... 
│   └── sampling_main
│       ├── prisma.png                            # PRISMA flowchart of sampling
│       ├── sample.xlsx                      # documentation of sampling process
│       └── top_journals.pdf                # screenshots of TOP database output
├── preregistration
│   └── preregistration.pdf                                    # preregistration
└── README.md                                                        # this file
```