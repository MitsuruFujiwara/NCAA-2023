#!/bin/sh
cd feats
rm *.feather
rm *.pkl
cd ../src

python 001_tourney_results.py
python 002_season_results.py
python 003_teams.py
python 004_seasons.py
python 005_seeds.py
python 006_massey_ordinals.py
python 007_coaches.py
python 008_conferences.py

python 101_aggregate.py

python 201_train_lgbm.py