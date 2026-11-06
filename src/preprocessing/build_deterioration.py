"""Build the deterioration cohort with static tables or time-series output.

Usage::

    # write the stay-level base and raw measurements (default)
    python -m src.preprocessing.build_deterioration

    # write the aggregated static table
    python -m src.preprocessing.build_deterioration --mode static

    # test the measurement stage using a saved checkpoint
    python -m src.preprocessing.build_deterioration --mode test
"""

import ast
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import pickle as pkl
import time
import datetime
from tqdm import tqdm as _tqdm
import os
import sys
from functools import partial
from sklearn.linear_model import LinearRegression

import argparse
from pathlib import Path

pd.options.mode.chained_assignment = None

_start_time = time.time()

# Show live progress only in a terminal. Set PROGRESS=1 to show occasional updates in logs.
_is_tty = sys.stderr.isatty()
_show_bar = _is_tty or os.environ.get("PROGRESS") == "1"
tqdm = partial(_tqdm, disable=not _show_bar, mininterval=1.0 if _is_tty else 60.0)

# ---------------------------------------------------------------------------
# TS mode (default) writes the stay-level base and raw measurements. Static mode
# writes the original aggregated summary table.
_parser = argparse.ArgumentParser(description="MDS-ED preprocessing (static or ts mode).")
_parser.add_argument("--mode", choices=["static", "ts", "test"], default="ts",
                     help="static: faithful mds_ed.csv reproduction (mds_ed_static.csv). "
                          "ts (default): stay-centric base + raw vital/lab time series "
                          "(mds_ed_base.csv, mds_ed_ts.csv, mds_ed_ts_keys.csv). "
                          "test: skip the ~45-min shared prefix, load the prefix checkpoint, "
                          "and run ONLY the ts vitals/labs stage (fast OOM/behaviour testing).")
_parser.add_argument("--no-checkpoint", dest="save_checkpoint", action="store_false",
                     help="do not write the prefix checkpoint in static/ts mode "
                          "(the checkpoint is what --mode test loads).")
_parser.add_argument("--stay-pick", choices=["first", "last"], default="last",
                     help="ts mode: when several ECGs map to one ED stay, which in-window "
                          "ECG seeds that stay's row in mds_ed_base.csv. 'last' (default) "
                          "keeps the latest ECG in [intime, intime+90min], 'first' the "
                          "earliest (the previous behaviour). Only affects the per-ECG-"
                          "varying static columns (biometrics_*, deterioration_mortality_stay).")
_args, _ = _parser.parse_known_args()
MODE = _args.mode
SAVE_CHECKPOINT = _args.save_checkpoint
STAY_PICK = _args.stay_pick
WINDOW_MINUTES = 90
print(f"Mode: {MODE}", flush=True)
if MODE in ('ts', 'test'):
    print(f"Stay pick (ts base): {STAY_PICK}", flush=True)
# ---------------------------------------------------------------------------

# Folder holding this repository beside data/, outputs/ and logs/
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
mimic_root = WORKSPACE_ROOT / "data" / "MIMIC-IV"
mimic_version = "3.1"
ed_version = "2.2"

hosp_dir = mimic_root / "mimiciv" / mimic_version / "hosp"
icu_dir = mimic_root / "mimiciv" / mimic_version / "icu"
ed_dir = mimic_root / "mimic-iv-ed" / ed_version / "ed"

output_dir = WORKSPACE_ROOT / "outputs" / "mds_ed"

# Checkpoint used by test mode to skip the shared setup stages
checkpoint_path = output_dir / "prefix_checkpoint.pkl"

records_path = ( mimic_root
    / "mimic-iv-ecg-ext-icd-diagnostic-labels-for-mimic-iv-ecg-1.0.1"
    / "records_w_diag_icd10.csv"
)

ed_stays_path = ed_dir / "edstays.csv.gz"
pyxis_path = ed_dir / "pyxis.csv.gz"
ed_diagnosis_path = ed_dir / "diagnosis.csv.gz"
vitals_path = ed_dir / "vitalsign.csv" #.gz"
triage_path = ed_dir / "triage.csv.gz"
admission_path = hosp_dir / "admissions.csv"#.gz"
procedure_path = hosp_dir / "procedures_icd.csv.gz"
diagnosis_path = hosp_dir / "diagnoses_icd.csv"#.gz"
omr_path = hosp_dir / "omr.csv"#.gz"
icu_stays_path = icu_dir / "icustays.csv"#.gz"
labitems_path = hosp_dir / "d_labitems.csv"#.gz"
labevents_path = hosp_dir / "labevents.csv"#.gz"

# Preprocess the tabular

if MODE != 'test':
    print("Building ECG-ED stay cohort (90-min window)...", flush=True)
    df_diags = pd.read_csv(records_path)
    df_diags['all_diag_all'] = df_diags['all_diag_all'].apply(ast.literal_eval)

    df_edstays = pd.read_csv(ed_stays_path) 
    df_diags = df_diags[df_diags['ed_stay_id'].isin(df_edstays['stay_id'].unique())]
    df_diags['ecg_time'] = pd.to_datetime(df_diags['ecg_time'])
    df_edstays['intime'] = pd.to_datetime(df_edstays['intime'])
    df_edstays['outtime'] = pd.to_datetime(df_edstays['outtime'])
    df_diags['dod'] = pd.to_datetime(df_diags['dod'])

    rows_to_keep = []

    # Iterate over each row in the df_diags DataFrame (unique MIMIC-ICD ECG rows)
    for _, row in tqdm(df_diags.iterrows(), total=len(df_diags), desc=" Cohort", leave=False):
        patient = row['subject_id']
        ecg_time = row['ecg_time']

        df_patient = df_edstays[df_edstays['subject_id'] == patient]
  
        for _, patient_row in df_patient.iterrows(): 
            intime = patient_row['intime']
            intime_90min = intime + pd.Timedelta(minutes=WINDOW_MINUTES)

            if intime <= ecg_time <= intime_90min:
                row_dict = row.to_dict()
                row_dict['stay_id'] = patient_row['stay_id']
                row_dict['intime'] = patient_row['intime']
                row_dict['outtime'] = patient_row['outtime']
                row_dict['gender'] = patient_row['gender']
                row_dict['race'] = patient_row['race']

                rows_to_keep.append(row_dict)
                break  

    df_filtered_diags = pd.DataFrame(rows_to_keep)
    df_filtered_diags.reset_index(drop=True, inplace=True)
    df_filtered_diags['90min'] = df_filtered_diags['intime'] + pd.Timedelta(minutes=WINDOW_MINUTES)

    # Mortality
    print("Mortality...", flush=True)
    df_filtered_diags['mortality_hours'] = (df_filtered_diags['dod'] - df_filtered_diags['intime']).dt.total_seconds() / 3600
    df_filtered_diags['mortality_days'] = (df_filtered_diags['dod'] - df_filtered_diags['intime']).dt.days


    mortality_cols = ['mortality_1D','mortality_7D','mortality_28D','mortality_90D','mortality_180D','mortality_365D']

    for col in mortality_cols:
        binary_values = []
        for _, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=f" Mortality {col}", leave=False):
        
            if pd.notnull(row['mortality_hours']) and row['mortality_hours'] < 1.5:
                binary_values.append(-999)
            else:
                mortality_days = row['mortality_days']
                ecg_time = row['ecg_time']
        
                if pd.notnull(mortality_days):
                    binary_values.append(1 if mortality_days <= int(col.split('_')[-1][:-1]) else 0)

                else:
                    binary_values.append(0)
                
        df_filtered_diags[col] = binary_values
    

    df_hosp = pd.read_csv(admission_path)
    dischtime_dict = df_hosp.set_index('hadm_id')['dischtime'].to_dict()
    df_filtered_diags['hosp_dischtime'] = df_filtered_diags['hosp_hadm_id'].map(dischtime_dict)
    df_filtered_diags['hosp_dischtime'] = pd.to_datetime(df_filtered_diags['hosp_dischtime'])

    condition = []

    for _, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" Mortality stay", leave=False):

        if pd.notnull(row['mortality_hours']) and row['mortality_hours'] < 1.5:
            condition.append(-999)
        
        elif not pd.notnull(row['dod']):
            condition.append(0)
        
        else:
            if row['dod']<row['hosp_dischtime']:
                condition.append(1)
            elif row['dod']<row['outtime']:
                condition.append(1)
            elif row['dod']>row['hosp_dischtime'] or row['dod']>row['outtime']:
                condition.append(0)


            
    df_filtered_diags['mortality_stay'] = condition
    mortality_cols.append('mortality_stay')

    df_filtered_diags[mortality_cols] = df_filtered_diags[mortality_cols].astype(float)


    # Diagnoses
    print("Diagnoses...", flush=True)
    df_filtered_diags = df_filtered_diags[df_filtered_diags['all_diag_all'].apply(lambda x: len(x) > 0)]
    df_filtered_diags["all_diag_all"]=df_filtered_diags["all_diag_all"].apply(lambda x: list(set([y.strip()[:5] for y in x])))
    df_filtered_diags["all_diag_all"]=df_filtered_diags["all_diag_all"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
    df_filtered_diags['all_diag_all'] = df_filtered_diags['all_diag_all'].apply(lambda code: code[:-1] if code[-1].isalpha() else code)

    def label_propagation(df, column_name, propagate_all=True):
        def flatten(l):
            return [item for sublist in l for item in sublist]
    
        col_flat = flatten(np.array(df[column_name]))
    
        def prepare_consistency_mapping_internal(codes_unique, codes_unique_all):
            res={}
            for c in codes_unique:
                if(propagate_all):
                    res[c]=[c[:i] for i in range(3,len(c)+1)]
                else: 
                    res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
            return res
    
        cons_map = prepare_consistency_mapping_internal(np.unique(col_flat), np.unique(col_flat))
        df[column_name] = df[column_name].apply(lambda x: list(set(flatten([cons_map[y] for y in x]))))
        return df

    df_filtered_diags = label_propagation(df_filtered_diags, "all_diag_all", propagate_all=True)


    def process_multilabel(df, column_name, threshold, output_column_name):

        counts = {}
    
        for row in df[column_name]:
            for item in row:
                counts[item] = counts.get(item, 0) + 1
    
        filtered_counts = {item: count for item, count in counts.items() if count >= threshold}
    
        unique_strings = sorted(filtered_counts, key=filtered_counts.get, reverse=True)
    
        df[output_column_name] = df[column_name].apply(lambda row: [1 if item in row else 0 for item in unique_strings])

        return df, np.array(unique_strings)

    df_filtered_diags, lbls_diags = process_multilabel(df_filtered_diags, 'all_diag_all', 230, 'Diagnoses_labels')



    # ICU

    print("ICU...", flush=True)
    df_icu = pd.read_csv(icu_stays_path)
    df_icu['intime'] = pd.to_datetime(df_icu['intime'])

    # Use every ICU stay so brief early contacts do not hide later admissions
    # ICU labels are based on the first contact and the stay duration thresholds
    hadm_to_icu_intimes = df_icu.groupby('hadm_id')['intime'].apply(list).to_dict()

    icus = []
    icu_24 = []
    icu_stay = []

    for i, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" ICU labels", leave=False):

        patient_intime = row['intime']
        intimes = hadm_to_icu_intimes.get(row['ed_hadm_id'])

        hrs = []
        if intimes:
            hrs = [(t - patient_intime).total_seconds() / 3600.0 for t in intimes]
            hrs = [h for h in hrs if not pd.isna(h)]

        if not hrs:                                   # no ICU stay (or only unusable NaT intimes)
            icus.append(np.nan)
            icu_24.append(0)
            icu_stay.append(0)
            continue

        icus.append(min(hrs))                         # time to first ICU contact
        qualifying = [h for h in hrs if h >= 1.5]     # ICU stays past the 1.5h floor

        if not qualifying:                            # every contact < 1.5h -> exclude (sentinel)
            icu_24.append(-999)
            icu_stay.append(-999)
        else:
            icu_stay.append(1)
            icu_24.append(1 if any(h <= 24 for h in qualifying) else 0)

    df_filtered_diags['icu_time_hours'] = icus
    df_filtered_diags['ICU_24H'] = icu_24
    df_filtered_diags['ICU_stay'] = icu_stay

    icu_cols = ['ICU_24H', 'ICU_stay']
    df_filtered_diags[icu_cols] = df_filtered_diags[icu_cols].astype(float)


    # targets with proedures

    print("Procedure targets (mechanical ventilation, ECMO)...", flush=True)
    df_proc = pd.read_csv(procedure_path)
    df_proc['chartdate'] = pd.to_datetime(df_proc['chartdate'])
    mechanical_ventilation_codes = ['9670', '9671', '9672', '5A1935Z', '5A1945Z', '5A1955Z'] 
    extracorporeal_membrane_oxygenation = ['3961', '3965', '3966', '5A1221Z', '5A1522G','5A1522H','5A15223','5A1522F','5A15A2F','5A15A2G','5A15A2H']  # ECMO codes

    new_col_mechanical_v = []
    new_col_ecmo = []

    for i, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" Procedures", leave=False):
        patient = row['subject_id']
        ed_hadm_id = row['ed_hadm_id']
        intime = row['intime']
    
        df_patient = df_proc[(df_proc['subject_id'] == patient) & (df_proc['hadm_id'] == ed_hadm_id)]
    
        df_patient_mechanical = df_patient[df_patient['icd_code'].isin(mechanical_ventilation_codes)]
        df_patient_mechanical = df_patient_mechanical[df_patient_mechanical['chartdate'].dt.date <= intime.date() + pd.Timedelta(days=1)]
        if not df_patient_mechanical.empty:
            new_col_mechanical_v.append(1)
        else:
            new_col_mechanical_v.append(0)
        
        df_patient_ecmo = df_patient[df_patient['icd_code'].isin(extracorporeal_membrane_oxygenation)]
        df_patient_ecmo = df_patient_ecmo[df_patient_ecmo['chartdate'].dt.date <= intime.date() + pd.Timedelta(days=1)]
        if not df_patient_ecmo.empty:
            new_col_ecmo.append(1)
        else:
            new_col_ecmo.append(0)
    

    df_filtered_diags['mechanical_ventilation'] = new_col_mechanical_v
    df_filtered_diags['ecmo'] = new_col_ecmo


    print("Cardiac arrest...", flush=True)
    d = pd.read_csv(ed_diagnosis_path)
    d_cardiac_ed = d[d['icd_title'].str.lower().str.contains('cardiac arrest')]

    cardiac_arrests = ['I469', '4275', 'I462', 'V1253', 'I468']

    d = pd.read_csv(diagnosis_path)
    d_cardiac_hosp = d[d['icd_code'].isin(cardiac_arrests)]
    d_cardiac_hosp = d_cardiac_hosp[d_cardiac_hosp['hadm_id'].isin(df_filtered_diags['hosp_hadm_id'])]

    cardiac_arrest = []

    for i, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" Cardiac arrest", leave=False):

        patient = row['subject_id']
        ed_stay_id = row['ed_stay_id']
        hosp_id = row['hosp_hadm_id']
    
        intime = row['intime']
    
        df_ed = d_cardiac_ed[(d_cardiac_ed['subject_id'] == patient) & (d_cardiac_ed['stay_id'] == ed_stay_id)]
    
        if len(df_ed)>0:
            ed_outtime = row['outtime']
            if ed_outtime <= intime+pd.Timedelta(days=1):
                cardiac_arrest.append(1)
            else:
                df_hosp = d_cardiac_hosp[(d_cardiac_hosp['subject_id'] == patient) & (d_cardiac_hosp['hadm_id'] == hosp_id)]
        
                if len(df_hosp)>0:
                    hosp_outtime = row['hosp_dischtime']
                    if hosp_outtime <= intime+pd.Timedelta(days=1):
                        cardiac_arrest.append(1)
            
                    else:
                        cardiac_arrest.append(0)
                    
                else:
                    cardiac_arrest.append(0)
        
        else:
            cardiac_arrest.append(0)
        
    df_filtered_diags['cardiac_arrest'] = cardiac_arrest

    vasopressors='epinephrine, norepinephrine, vasopressin, dobutamine, dopamine, or phenylephrine'
    inotropes='epinephrine, dobutamine, or dopamine'


    print("Medications (vasopressors, inotropes)...", flush=True)
    df_pyxis = pd.read_csv(pyxis_path)
    df_pyxis = df_pyxis[df_pyxis['subject_id'].isin(df_filtered_diags['subject_id'].unique())]
    df_pyxis = df_pyxis.dropna(subset='gsn')
    df_pyxis['charttime'] = pd.to_datetime(df_pyxis['charttime'])

    total_vasopressors = ['Norepinephrine','NORepinephrine','EPINEPHrine (for dilution)',
                          'NORepinephrine (in NS)','EPINEPHrine Auto Injector','EPINEPHrine (for dil 2mg/2mL 2mL',
                          'EPINEPHrine','EPINEPHrine 1mg/1mL 1mL AMP','EPINEPHrine (B 1mg/10mL 10mL SYR',
                          'EPINEPHrine (Bristojet)','EPINEPHrine 1mg/10mL 10mL SYR','NORepinephrine ((for dil 8mg KIT',
                          'Norepinephrine','NORepinephrine','NORepinephrine (in NS)',
                          'NORepinephrine ((for dil 8mg KIT','Vasopressin','Vasopressin (for 40units/2mL 2mL',
                          'Vasopressin (for dilution)','DOPamine','DOPamine 400mg BAG',
                          'PHENYLEPHrine','PHENYLEPHrine (for dilution)','Phenylephrine',
                          'PHENYLEPHrine (for dilu 60mg/6mL','PHENYLEPHrine (for 50mg/5mL VIAL','PHENYLEPHrine (QuVa)',
                          'DOBUTamine 250mg BAG','DOBUTamine']

    total_inotropes = ['EPINEPHrine (for dilution)','EPINEPHrine Auto Injector','EPINEPHrine (for dil 2mg/2mL 2mL',
                       'EPINEPHrine','EPINEPHrine 1mg/1mL 1mL AMP','EPINEPHrine (B 1mg/10mL 10mL SYR',
                       'EPINEPHrine (Bristojet)','EPINEPHrine 1mg/10mL 10mL SYR','DOPamine',
                       'DOPamine 400mg BAG','DOBUTamine 250mg BAG','DOBUTamine']

    vasopressors = []
    inotropes = []

    for i, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" Medications", leave=False):

        stay_id = row['ed_stay_id']
        intime = row['intime']
    
        df_meds = df_pyxis[df_pyxis['stay_id'] == stay_id]
    
        cutoff_time = row['90min']
    
        meds = df_meds[df_meds['charttime'] >= cutoff_time]['name'].tolist()
    
        if any(med in total_vasopressors for med in meds):
            vasopressors.append(1)
        else:
            vasopressors.append(0)

        if any(med in total_inotropes for med in meds):
            inotropes.append(1)
        else:
            inotropes.append(0)
        
    df_filtered_diags['vasopressors'] = vasopressors
    df_filtered_diags['inotropes'] = inotropes


    print("Severe hypoxemia...", flush=True)
    df_vitalsign = pd.read_csv(vitals_path)
    df_vitalsign = df_vitalsign[df_vitalsign['subject_id'].isin(df_filtered_diags['subject_id'].unique())]
    df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

    df_vitalsign.loc[df_vitalsign['temperature'] < 50, 'temperature'] = np.nan # 53 minimum recorded
    df_vitalsign.loc[df_vitalsign['temperature'] > 150, 'temperature'] = np.nan # 115.7 maximum recorded
    df_vitalsign.loc[df_vitalsign['heartrate'] > 700, 'heartrate'] = np.nan # 600 maximum recorded
    df_vitalsign.loc[df_vitalsign['resprate'] > 300, 'resprate'] = np.nan # normal 20, athletes 50
    df_vitalsign.loc[df_vitalsign['o2sat'] > 100, 'o2sat'] = np.nan # can't be negative nor more than 100
    df_vitalsign.loc[df_vitalsign['o2sat'] < 0, 'o2sat'] = np.nan # can't be negative nor more than 100
    df_vitalsign.loc[df_vitalsign['dbp'] > 500, 'dbp'] = np.nan # max recorded 370
    df_vitalsign.loc[df_vitalsign['sbp'] > 500, 'sbp'] = np.nan # max recorded 360


    severe_hypoxemia = []

    for i, row in tqdm(df_filtered_diags.iterrows(), total=len(df_filtered_diags), desc=" Severe hypoxemia", leave=False):

        df_patient_vital_before = df_vitalsign[(df_vitalsign['stay_id'] == row['ed_stay_id']) &
                                        (df_vitalsign['charttime'] <= row['90min'])]
        df_patient_vital_before = df_patient_vital_before.dropna(subset=['o2sat'])
    
    
        df_patient_vital_after = df_vitalsign[(df_vitalsign['stay_id'] == row['ed_stay_id']) & 
                                        (df_vitalsign['charttime'] > row['90min'])]
        df_patient_vital_after = df_patient_vital_after.dropna(subset=['o2sat'])
    
        if len(df_patient_vital_after)==0: 
            severe_hypoxemia.append(-999)
    
        elif len(df_patient_vital_before)==0: 
            severe_hypoxemia.append(-999)
    
        elif any(df_patient_vital_before['o2sat']<=85): 
            severe_hypoxemia.append(-999)
    
        else:
        
            df_patient_vital_within = df_patient_vital_after[df_patient_vital_after['charttime']<= row['intime']+pd.Timedelta(days=1)]
            df_patient_vital_after = df_patient_vital_after[df_patient_vital_after['charttime']> row['intime']+pd.Timedelta(days=1)]
        
            if any(df_patient_vital_within['o2sat']<=85):
                severe_hypoxemia.append(1)

            elif not any(df_patient_vital_within['o2sat']<=85) and not any(df_patient_vital_after['o2sat']<=85):
                severe_hypoxemia.append(0)
            
            elif len(df_patient_vital_within)>0 and not any(df_patient_vital_within['o2sat']<=85) and any(df_patient_vital_after['o2sat']<=85): 
                severe_hypoxemia.append(0)
        
            elif len(df_patient_vital_within)==0 and not any(df_patient_vital_within['o2sat']<=85) and any(df_patient_vital_after['o2sat']<=85): 
                severe_hypoxemia.append(-999)
            
    df_filtered_diags['severe_hypoxemia'] = severe_hypoxemia
    decompensation_cols = ['severe_hypoxemia','ecmo','vasopressors','inotropes', 'mechanical_ventilation','cardiac_arrest']
    df_features = df_filtered_diags.copy()

    df_features.rename(columns={'Unnamed: 0': 'data'}, inplace=True)
    decompensation_model_cols = decompensation_cols + icu_cols + mortality_cols
    df_features['Deterioration_labels'] = df_features[decompensation_model_cols].values.tolist()
    df_features.drop(columns=decompensation_model_cols, inplace=True)
    df_features.drop(columns=['fold'], inplace=True)
    # labels_cols = ['Diagnoses_labels','Deterioration_labels']


    # features

    print("Demographics...", flush=True)
    df_features['gender'] = df_features['gender'].apply(lambda x: 1 if x=='M' else 0)
    def map_race(row):
        if 'WHITE' in row:
            return 'white'
        elif 'BLACK' in row:
            return 'black/african'
        elif 'HISPANIC' in row:
            return 'hispanic/latino'
        elif 'ASIAN' in row:
            return 'asian'
        else:
            return 'other'
    
    df_features['race_mapped'] = df_features['race'].apply(map_race)

    race_dummies = pd.get_dummies(df_features['race_mapped'], prefix='demographics_ethnicity')

    df_features = pd.concat([df_features, race_dummies], axis=1)


    # biometrics

    print("Biometrics...", flush=True)
    omr = pd.read_csv(omr_path)
    omr['result_value'] = pd.to_numeric(omr['result_value'], errors='coerce')
    omr['chartdate'] = pd.to_datetime(omr['chartdate'])
    omr = omr[omr['result_name'].isin(['BMI (kg/m2)','Height (Inches)','Weight (Lbs)'])]

    omr = omr[omr['subject_id'].isin(df_features['subject_id'].unique())]
    omr.dropna(subset=['result_value'], inplace=True)
    omr.loc[omr['result_name'] == 'Height (Inches)', 'result_value'] *= 2.54
    omr.loc[omr['result_name'] == 'Height (Inches)', 'result_name'] = 'Height (cm)'
    omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_value'] *= 0.453592
    omr.loc[omr['result_name'] == 'Weight (Lbs)', 'result_name'] = 'Weight (kg)'

    conditions = [
        (omr['result_name'] == 'BMI (kg/m2)') & (omr['result_value'] > 100),
        (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] > 400),
        (omr['result_name'] == 'Height (cm)') & (omr['result_value'] > 400),
        (omr['result_name'] == 'Weight (kg)') & (omr['result_value'] < 20),
        (omr['result_name'] == 'Height (cm)') & (omr['result_value'] < 60)
    ]

    for condition in conditions:
        omr.loc[condition, 'result_value'] = np.nan
    
    omr.dropna(subset=['result_value'], inplace=True)


    out_bmi = []
    out_weight = []
    out_height = []

    for _, row in tqdm(df_features.iterrows(), total=len(df_features), leave=False):
        patient = row['subject_id']
        intime = row['ecg_time']
    
        df_patient = omr[omr['subject_id'] == patient]
        df_patient_within = df_patient.loc[(df_patient['chartdate'] >= (intime - pd.Timedelta(days=30))) & 
                                           (df_patient['chartdate'] <= (intime + pd.Timedelta(days=30)))]
    
        if df_patient_within.empty:
            out_bmi.append(np.nan)
            out_weight.append(np.nan)
            out_height.append(np.nan)
        else:
            # Find the closest BMI to ecg_time
            bmi_rows = df_patient_within[df_patient_within['result_name'] == 'BMI (kg/m2)']
            if not bmi_rows.empty:
                closest_bmi = bmi_rows.iloc[(bmi_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
                out_bmi.append(closest_bmi)
            else:
                out_bmi.append(np.nan)
        
            # Find the closest Weight to ecg_time
            weight_rows = df_patient_within[df_patient_within['result_name'] == 'Weight (kg)']
            if not weight_rows.empty:
                closest_weight = weight_rows.iloc[(weight_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
                out_weight.append(closest_weight)
            else:
                out_weight.append(np.nan)
        
            # Find the closest Height to ecg_time
            height_rows = df_patient_within[df_patient_within['result_name'] == 'Height (cm)']
            if not height_rows.empty:
                closest_height = height_rows.iloc[(height_rows['chartdate'] - intime).abs().argsort()[:1]]['result_value'].values[0]
                out_height.append(closest_height)
            else:
                out_height.append(np.nan)
            
    df_features['biometrics_bmi'] = out_bmi
    df_features['biometrics_weight'] = out_weight
    df_features['biometrics_height'] = out_height


    # vital signs

    print("Vital sign features...", flush=True)
    df_vitalsign = pd.read_csv(vitals_path)
    df_vitalsign = df_vitalsign[df_vitalsign['subject_id'].isin(df_features['subject_id'].unique())]
    df_vitalsign['charttime'] = pd.to_datetime(df_vitalsign['charttime'])

    df_vitalsign.loc[df_vitalsign['temperature'] < 50, 'temperature'] = np.nan # 53 minimum recorded
    df_vitalsign.loc[df_vitalsign['temperature'] > 150, 'temperature'] = np.nan # 115.7 maximum recorded
    df_vitalsign.loc[df_vitalsign['heartrate'] > 700, 'heartrate'] = np.nan # 600 maximum recorded
    df_vitalsign.loc[df_vitalsign['resprate'] > 300, 'resprate'] = np.nan # normal 20, athletes 50
    df_vitalsign.loc[df_vitalsign['o2sat'] > 100, 'o2sat'] = np.nan # can't be negative nor more than 100
    df_vitalsign.loc[df_vitalsign['o2sat'] < 0, 'o2sat'] = np.nan # can't be negative nor more than 100
    df_vitalsign.loc[df_vitalsign['dbp'] > 500, 'dbp'] = np.nan # max recorded 370
    df_vitalsign.loc[df_vitalsign['sbp'] > 500, 'sbp'] = np.nan # max recorded 360

    def fahrenheit_to_celsius(f):
        return (f - 32) * 5.0/9.0

    df_vitalsign['temperature'] = df_vitalsign['temperature'].apply(fahrenheit_to_celsius)

    new_df_vital = []

    for stay_id in tqdm(df_vitalsign['stay_id'].unique(), desc=" Merging vital signs with features", leave=False):

        df_vital_stay = df_vitalsign[df_vitalsign['stay_id'] == stay_id]
        df_intime = df_features[df_features['ed_stay_id'] == stay_id]
    
        if len(df_intime)>0:
    
            intime = df_intime['intime'].values[0]
        
            df_vital_stay_filtered = df_vital_stay[
            (df_vital_stay['charttime'] >= intime) &
            (df_vital_stay['charttime'] <= intime + pd.Timedelta(minutes=WINDOW_MINUTES))]
        
            if len(df_vital_stay_filtered)>0:
                new_df_vital.append(df_vital_stay_filtered)
            
    df_vital = pd.concat(new_df_vital)

    intime_dict = df_features.set_index('ed_stay_id')['intime'].to_dict()
    df_vital['intime'] = df_vital['stay_id'].map(intime_dict)
    df_vital['minutes_since_intime'] = (df_vital['charttime']-df_vital['intime']).dt.total_seconds() / 60

    # Save the shared data so test mode can skip the expensive early stages
    if SAVE_CHECKPOINT:
        print(f'Saving prefix checkpoint -> {checkpoint_path} ...', flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'wb') as _ckpt_f:
            pkl.dump({'df_features': df_features, 'df_vital': df_vital,
                      'lbls_diags': lbls_diags,
                      'decompensation_model_cols': decompensation_model_cols},
                     _ckpt_f, protocol=pkl.HIGHEST_PROTOCOL)
        print(f'Checkpoint saved ({checkpoint_path.stat().st_size/1e6:.1f} MB).', flush=True)


# Shared processing for static and time-series output modes

vp_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

labs_to_keep = ['Absolute Basophil Count', 'Absolute Eosinophil Count',
       'Absolute Lymphocyte Count', 'Alanine Aminotransferase (ALT)',
       'Albumin', 'Alkaline Phosphatase',
       'Asparate Aminotransferase (AST)', 'Bands', 'Base Excess',
       'Basophils', 'Bicarbonate', 'Bilirubin, Direct',
       'Bilirubin, Total', 'C-Reactive Protein', 'Calcium, Total',
       'Carboxyhemoglobin', 'Chloride', 'Creatine Kinase (CK)',
       'Creatine Kinase, MB Isoenzyme', 'Creatinine', 'Eosinophils',
       'Fibrinogen, Functional', 'Free Calcium', 'Glucose', 'Hematocrit',
       'Hemoglobin', 'INR(PT)', 'Lactate', 'Lymphocytes', 'Magnesium',
       'Neutrophils', 'Oxygen Saturation',
       'PT', 'PTT', 'Phosphate', 'Platelet Count', 'Potassium', 'RDW',
       'Red Blood Cells', 'Sodium',
        'Troponin T', 'Urea Nitrogen','White Blood Cells', 'pCO2', 'pH']

# Upper limits for each laboratory test
lab_upper_bounds = {
    'Absolute Basophil Count': 20, 'Absolute Eosinophil Count': 20,
    'Absolute Lymphocyte Count': 100, 'Alanine Aminotransferase (ALT)': 2000,
    'Alkaline Phosphatase': 2000, 'Asparate Aminotransferase (AST)': 2000,
    'Creatine Kinase (CK)': 2000, 'Glucose': 2000, 'Lactate': 2000,
    'Platelet Count': 2000,
}
lab_stats = ['mean', 'median', 'min', 'max', 'std', 'first', 'last', 'rate_change', 'coeff']


def add_triage_acuity(df_features):
    """Add one triage score for each emergency department stay"""
    print("Triage acuity...", flush=True)
    acuities = []

    df_triage = pd.read_csv(triage_path)

    for i, row in tqdm(df_features.iterrows(), total=len(df_features), desc=" Triage acuity", leave=False):
        row_acuity = df_triage[df_triage['stay_id'] == row['ed_stay_id']]
        if len(row_acuity) > 0:
            acu = row_acuity['acuity'].values[0]
            acuities.append(acu)
        else:
            acuities.append(np.nan)

    df_features['vitals_acuity'] = acuities


def build_lab_window(df_features):
    """Read laboratory measurements from the first 90 minutes of each stay."""
    print("Lab values...", flush=True)
    # Map each laboratory test to its name and specimen type
    df_labtitles = pd.read_csv(labitems_path)
    id2label = df_labtitles.set_index('itemid')['label'].to_dict()
    id2fluid = df_labtitles.set_index('itemid')['fluid'].to_dict()
    kept_itemids = df_labtitles.loc[df_labtitles['label'].isin(labs_to_keep), 'itemid'].tolist()

    # Convert the large laboratory file to a small cached table once
    output_dir.mkdir(parents=True, exist_ok=True)
    labevents_parquet = output_dir / 'labevents_subset.parquet'
    if not labevents_parquet.exists():
        print(" Converting labevents.csv -> parquet (one-time)...", flush=True)
        _read_opts = pacsv.ReadOptions(block_size=1 << 26)  # 64 MB blocks
        _conv_opts = pacsv.ConvertOptions(
            include_columns=['subject_id', 'itemid', 'charttime', 'valuenum'],
            column_types={'subject_id': pa.int64(), 'itemid': pa.int64(),
                          'valuenum': pa.float64(), 'charttime': pa.timestamp('us')},
            strings_can_be_null=True,
        )
        _reader = pacsv.open_csv(labevents_path, read_options=_read_opts, convert_options=_conv_opts)
        _writer = None
        try:
            for _batch in _reader:
                if _writer is None:
                    _writer = pq.ParquetWriter(labevents_parquet, _batch.schema)
                _writer.write_batch(_batch)
        finally:
            if _writer is not None:
                _writer.close()

    cohort_subject_ids = df_features['subject_id'].unique().tolist()

    # Keep the most common specimen type for each test
    print(" Selecting most-common specimen fluid per lab...", flush=True)
    fluid_counts = (
        pl.scan_parquet(labevents_parquet)
          .filter(pl.col('itemid').is_in(kept_itemids))
          .filter(pl.col('subject_id').is_in(cohort_subject_ids))
          .group_by('itemid').agg(pl.len().alias('n'))
          .collect(streaming=True)
          .to_pandas()
    )
    fluid_counts['label'] = fluid_counts['itemid'].map(id2label)
    fluid_counts['fluid'] = fluid_counts['itemid'].map(id2fluid)
    per_fluid = fluid_counts.groupby(['label', 'fluid'])['n'].sum().reset_index()
    winning_fluid = (per_fluid.loc[per_fluid.groupby('label')['n'].idxmax()]
                     .set_index('label')['fluid'].to_dict())

    # Save the selected specimen type for review
    fluid_table = per_fluid.sort_values(['label', 'n'], ascending=[True, False]).copy()
    fluid_table['selected'] = [winning_fluid.get(lab) == fl
                               for lab, fl in zip(fluid_table['label'], fluid_table['fluid'])]
    fluid_table_path = output_dir / 'mds_ed_fluid_selection.csv'
    fluid_table.to_csv(fluid_table_path, index=False)
    _non_blood = fluid_table[fluid_table['selected'] & (fluid_table['fluid'] != 'Blood')]
    print(f"  Wrote specimen-fluid selection table -> {fluid_table_path} "
          f"({fluid_table['label'].nunique()} labs; "
          f"{len(_non_blood)} non-Blood winner(s)"
          + (": " + ", ".join(f"{r.label}={r.fluid}" for _, r in _non_blood.iterrows())
             if len(_non_blood) else "") + ")", flush=True)

    allowed_itemids = [iid for iid in kept_itemids
                       if id2label.get(iid) not in winning_fluid
                       or id2fluid.get(iid) == winning_fluid[id2label[iid]]]

    # Keep the original row order so measurements map back to each stay
    cohort = df_features[['subject_id', 'intime', '90min']].rename(columns={'90min': 'cutoff'}).copy()
    cohort['feat_idx'] = np.arange(len(cohort))
    cohort_pl = pl.from_pandas(cohort).with_columns(
        pl.col('intime').cast(pl.Datetime('us')),
        pl.col('cutoff').cast(pl.Datetime('us')),
    )

    # Keep measurements from the first 90 minutes of each stay
    print(" Gathering in-window lab measurements...", flush=True)
    df_win = (
        pl.scan_parquet(labevents_parquet)
          .filter(pl.col('itemid').is_in(allowed_itemids))
          .join(cohort_pl.lazy(), on='subject_id', how='inner')
          .filter((pl.col('charttime') >= pl.col('intime')) & (pl.col('charttime') <= pl.col('cutoff')))
          .with_columns(
              ((pl.col('charttime') - pl.col('intime')).dt.total_seconds() / 60).alias('minutes_since_intime')
          )
          .select(['feat_idx', 'itemid', 'valuenum', 'charttime', 'minutes_since_intime'])
          .collect(streaming=True)
          .to_pandas()
    )

    df_win['label'] = df_win['itemid'].map(id2label)
    # Remove values above each test's allowed limit and duplicate rows
    for lab_name, bound in lab_upper_bounds.items():
        m = (df_win['label'] == lab_name) & (df_win['valuenum'] > bound)
        df_win.loc[m, 'valuenum'] = np.nan
    df_win = df_win.drop_duplicates(subset=['feat_idx', 'itemid', 'minutes_since_intime', 'valuenum'])
    df_win = df_win.dropna(subset=['valuenum'])
    return df_win


def finalise_features(df_features, lbls_diags, lbls_det):
    """Prepare shared features before writing the selected output files."""
    print("Finalising (label expansion, renaming)...", flush=True)
    df_features.drop(columns='race_mapped', inplace=True)
    df_features.rename(columns={'age':'demographics_age','gender':'demographics_gender'}, inplace=True)

    columns_to_rename = [
        'data', 'file_name', 'study_id', 'subject_id', 'ecg_time',
        'ed_stay_id', 'ed_hadm_id', 'hosp_hadm_id', 'ed_diag_ed',
        'ed_diag_hosp', 'hosp_diag_hosp', 'all_diag_hosp', 'all_diag_all',
        'anchor_year', 'anchor_age', 'dod', 'ecg_no_within_stay',
        'ecg_taken_in_ed', 'ecg_taken_in_hosp', 'ecg_taken_in_ed_or_hosp',
        'strat_fold', 'stay_id', 'intime', 'outtime', 'race', '90min',
        'mortality_hours', 'mortality_days', 'hosp_dischtime', 'icu_time_hours'
    ]
    renaming_dict = {col: f'general_{col}' for col in columns_to_rename}
    df_features.rename(columns=renaming_dict, inplace=True)

    # Expand diagnosis labels together to reduce memory use
    diag_cols = [f'diagnoses_{label}' for label in lbls_diags]
    det_cols = [f'deterioration_{label}' for label in lbls_det]
    diag_expanded = pd.DataFrame(
        df_features['Diagnoses_labels'].tolist(),
        columns=diag_cols, index=df_features.index, dtype='int8',
    )
    # Keep deterioration labels as floats because some use -999 as a missing value
    det_expanded = pd.DataFrame(
        df_features['Deterioration_labels'].tolist(),
        columns=det_cols, index=df_features.index,
    )
    df_features = df_features.drop(columns=['Diagnoses_labels', 'Deterioration_labels'])
    df_features = pd.concat([df_features, diag_expanded, det_expanded], axis=1)

    columns_to_drop = [
        'general_hosp_hadm_id',
        'general_hosp_diag_hosp',
        'general_all_diag_hosp',
        'general_all_diag_all',
        'general_ecg_taken_in_ed',
        'general_ecg_taken_in_hosp',
        'general_ecg_taken_in_ed_or_hosp',
        'general_stay_id'
    ]

    df_diags = pd.read_csv(records_path)
    df_diags['general_data'] = df_diags.index
    df_diags['general_study_id'] = df_diags['study_id']

    df_features['general_data'] = df_features['general_study_id'].map(df_diags.set_index('general_study_id')['general_data'])
    df_features.drop(columns=columns_to_drop, inplace=True) # Remove unused fields
    df_features.columns = df_features.columns.str.replace(' ', '_', regex=False)
    df_features.columns = df_features.columns.str.lower()
    return df_features


def save_ts_outputs(df_features, df_vital, df_win):
    """Write time-series files with one record for each emergency department stay."""
    print("Building stay-centric time-series outputs (ts mode)...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_col = 'minutes_since_intime'

    # Create one index row for each stay
    idx = (df_features[['subject_id', 'ed_stay_id', 'ed_hadm_id', 'intime', '90min']]
           .rename(columns={'90min': 'cutoff'})
           .drop_duplicates('ed_stay_id', keep='first')
           .reset_index(drop=True))
    subj_by_stay = idx.set_index('ed_stay_id')['subject_id']
    hadm_by_stay = idx.set_index('ed_stay_id')['ed_hadm_id']
    intime_by_stay = idx.set_index('ed_stay_id')['intime']
    cutoff_by_stay = idx.set_index('ed_stay_id')['cutoff']

    # Keep the first vital reading at each stay and time
    v = df_vital.rename(columns={'stay_id': 'ed_stay_id'})
    v = v[['ed_stay_id', time_col, 'charttime'] + vp_cols]
    vitals_ts = (v.sort_values(['ed_stay_id', time_col])
                  .groupby(['ed_stay_id', time_col], as_index=False, sort=False)
                  .agg({c: 'first' for c in ['charttime', *vp_cols]}))

    # Map laboratory measurements to stays and remove duplicates
    stay_by_featidx = df_features['ed_stay_id'].to_numpy()
    lab = df_win.copy()
    lab['ed_stay_id'] = stay_by_featidx[lab['feat_idx'].to_numpy()]
    lab = lab.drop_duplicates(['ed_stay_id', 'itemid', 'charttime', 'valuenum', time_col])
    if not lab.empty:
        labs_ts = (lab.pivot_table(index=['ed_stay_id', time_col], columns='label',
                                   values='valuenum', aggfunc='first')
                      .reset_index()
                      .sort_values(['ed_stay_id', time_col]))
        labs_ts.columns.name = None
        time_info = lab.groupby(['ed_stay_id', time_col], as_index=False)[['charttime']].first()
        labs_ts = labs_ts.merge(time_info, on=['ed_stay_id', time_col], how='left')
    else:
        labs_ts = pd.DataFrame(columns=['ed_stay_id', time_col, 'charttime'])
    lab_feature_cols = [c for c in labs_ts.columns if c not in ('ed_stay_id', time_col, 'charttime')]

    # Combine vital signs and laboratory measurements by stay and time
    combined_ts = (pd.merge(vitals_ts, labs_ts, on=['ed_stay_id', time_col], how='outer',
                            suffixes=('_v', '_l'))
                     .sort_values(['ed_stay_id', time_col])
                     .reset_index(drop=True))
    combined_ts['charttime'] = combined_ts['charttime_v'].fillna(combined_ts['charttime_l'])
    combined_ts.drop(columns=['charttime_v', 'charttime_l'], inplace=True)
    combined_ts['subject_id'] = combined_ts['ed_stay_id'].map(subj_by_stay)
    combined_ts['ed_hadm_id'] = combined_ts['ed_stay_id'].map(hadm_by_stay)
    combined_ts['general_intime'] = combined_ts['ed_stay_id'].map(intime_by_stay)
    combined_ts['general_90min'] = combined_ts['ed_stay_id'].map(cutoff_by_stay)

    front = ['ed_stay_id', 'subject_id', 'ed_hadm_id',
             'general_intime', 'general_90min', 'charttime', time_col]
    feature_cols_ordered = list(vp_cols) + lab_feature_cols
    # Use the same feature names as the static output
    feature_rename = {f: f"vitals_{f}" for f in vp_cols}
    feature_rename.update({lab_name: f"labvalues_{str(lab_name).lower().replace(' ', '_')}"
                           for lab_name in lab_feature_cols})
    combined_ts = combined_ts[front + feature_cols_ordered].rename(columns=feature_rename)
    renamed_feature_cols = [feature_rename[c] for c in feature_cols_ordered]
    for col in ('ed_stay_id', 'subject_id', 'ed_hadm_id'):
        combined_ts[col] = combined_ts[col].astype('Int64')

    # Keep one static row per stay using the configured ECG selection
    base = finalise_features(df_features, lbls_diags, decompensation_model_cols).reset_index(drop=True)
    base = (base.sort_values('general_ecg_time', kind='stable', na_position='first')
                .drop_duplicates('general_ed_stay_id', keep=STAY_PICK)
                .sort_index()
                .reset_index(drop=True))
    ecg_cols = ['general_data', 'general_file_name', 'general_study_id',
                'general_ecg_time', 'general_ecg_no_within_stay']
    base = base.drop(columns=[c for c in ecg_cols if c in base.columns])

    base.to_csv(output_dir / 'mds_ed_base.csv', index=False)
    combined_ts.to_csv(output_dir / 'mds_ed_ts.csv', index=False)

    # Write a summary of available measurements for each stay
    vitals_present = set(vitals_ts['ed_stay_id'].unique())
    labs_present = set(labs_ts['ed_stay_id'].unique()) if len(labs_ts) else set()
    key_columns = ['ed_stay_id', 'subject_id', 'ed_hadm_id',
                   'n_timepoints', 'n_features', 'has_ts', 'has_vitals', 'has_labvalues']
    if len(combined_ts):
        grp = combined_ts.groupby('ed_stay_id', sort=False)
        keys_df = grp[['subject_id', 'ed_hadm_id']].first()
        keys_df['n_timepoints'] = grp[time_col].nunique()
        present = combined_ts[renamed_feature_cols].notna()
        present.insert(0, 'ed_stay_id', combined_ts['ed_stay_id'].to_numpy())
        keys_df['n_features'] = present.groupby('ed_stay_id', sort=False).any().sum(axis=1)
        keys_df = keys_df.reset_index()
        keys_df['has_vitals'] = keys_df['ed_stay_id'].isin(vitals_present).astype(int)
        keys_df['has_labvalues'] = keys_df['ed_stay_id'].isin(labs_present).astype(int)
        keys_df['has_ts'] = (keys_df['has_vitals'] | keys_df['has_labvalues']).astype(int)
        keys_df = keys_df[key_columns]
        for col in ('subject_id', 'ed_stay_id', 'ed_hadm_id'):
            keys_df[col] = keys_df[col].astype('Int64')
    else:
        keys_df = pd.DataFrame(columns=key_columns)
    keys_df.to_csv(output_dir / 'mds_ed_ts_keys.csv', index=False)

    print("Done (ts mode). Wrote:", flush=True)
    print(f"  {output_dir / 'mds_ed_base.csv'} ({len(base)} rows, {len(base.columns)} cols, "
          f"{base['general_ed_stay_id'].nunique()} stays)", flush=True)
    print(f"  {output_dir / 'mds_ed_ts.csv'} ({combined_ts['ed_stay_id'].nunique()} stays, "
          f"{len(combined_ts)} rows, {len(vp_cols) + len(lab_feature_cols)} features)", flush=True)
    print(f"  {output_dir / 'mds_ed_ts_keys.csv'} ({len(keys_df)} stays)", flush=True)


if MODE == 'test':
    print(f"[test] loading prefix checkpoint {checkpoint_path} ...", flush=True)
    if not checkpoint_path.exists():
        raise SystemExit(f"[test] checkpoint not found: {checkpoint_path}. "
                         f"Run once in ts (or static) mode first to create it.")
    with open(checkpoint_path, 'rb') as _ckpt_f:
        _ckpt = pkl.load(_ckpt_f)
    df_features = _ckpt['df_features']
    df_vital = _ckpt['df_vital']
    lbls_diags = _ckpt['lbls_diags']
    decompensation_model_cols = _ckpt['decompensation_model_cols']
    print(f"[test] loaded df_features={df_features.shape}, df_vital={df_vital.shape}, "
          f"{len(lbls_diags)} diagnosis labels", flush=True)

if MODE in ('ts', 'test'):
    add_triage_acuity(df_features)
    df_win = build_lab_window(df_features) 
    save_ts_outputs(df_features, df_vital, df_win)
    print(f"Total time: {datetime.timedelta(seconds=round(time.time() - _start_time))}", flush=True)
    raise SystemExit(0)


# ===== STATIC MODE (original pipeline, unchanged) =====

vp_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']

vital_features = []

for i, row in tqdm(df_features.iterrows(), total=len(df_features), desc=" Aggregating vital sign features", leave=False):
    df_vital_p = df_vital[df_vital['stay_id'] == row['ed_stay_id']]
    
    if len(df_vital_p) > 0:
        features = {}
        for col in vp_cols:
            vals = df_vital_p[col].dropna().values  # Drop NaNs for computation
            df_vital_pv = df_vital_p.dropna(subset=col)
            if len(vals) > 0:
                mean_val = np.mean(vals)
                median_val = np.median(vals)
                min_val = np.min(vals)
                max_val = np.max(vals)
                std_val = np.std(vals)
                first_val = vals[0]
                last_val = vals[-1]
                
                rate_change = (last_val - first_val) / first_val if first_val != 0 else np.nan
                
                X = df_vital_pv['minutes_since_intime'].values.reshape(-1, 1)
                y = vals.reshape(-1, 1)
                model = LinearRegression().fit(X, y)
                coeff = model.coef_[0][0]
                
                # Store computed features
                features[f"vitals_{col}_mean"] = mean_val
                features[f"vitals_{col}_median"] = median_val
                features[f"vitals_{col}_min"] = min_val
                features[f"vitals_{col}_max"] = max_val
                features[f"vitals_{col}_std"] = std_val
                features[f"vitals_{col}_first"] = first_val
                features[f"vitals_{col}_last"] = last_val
                features[f"vitals_{col}_rate_change"] = rate_change
                features[f"vitals_{col}_coeff"] = coeff
            else:
                # If no values, set all features to NaN
                features[f"vitals_{col}_mean"] = np.nan
                features[f"vitals_{col}_median"] = np.nan
                features[f"vitals_{col}_min"] = np.nan
                features[f"vitals_{col}_max"] = np.nan
                features[f"vitals_{col}_std"] = np.nan
                features[f"vitals_{col}_first"] = np.nan
                features[f"vitals_{col}_last"] = np.nan
                features[f"vitals_{col}_rate_change"] = np.nan
                features[f"vitals_{col}_coeff"] = np.nan
        
        # Append features to vital_features
        vital_features.append(features)
    
    else:
        # If no vital sign data for the stay_id, append NaN for all features
        features = {}
        for col in vp_cols:
            features[f"vitals_{col}_mean"] = np.nan
            features[f"vitals_{col}_median"] = np.nan
            features[f"vitals_{col}_min"] = np.nan
            features[f"vitals_{col}_max"] = np.nan
            features[f"vitals_{col}_std"] = np.nan
            features[f"vitals_{col}_first"] = np.nan
            features[f"vitals_{col}_last"] = np.nan
            features[f"vitals_{col}_rate_change"] = np.nan
            features[f"vitals_{col}_coeff"] = np.nan
            
        vital_features.append(features)
        
vital_features_df = pd.DataFrame(vital_features)
df_features.reset_index(inplace=True, drop=True)

df_features = pd.concat([df_features, vital_features_df], axis=1)

add_triage_acuity(df_features)


df_win = build_lab_window(df_features)

# Calculate summary values for each stay and laboratory test
print(" Computing lab summary statistics...", flush=True)
full_cols = [f"labvalues_{col}_{stat}" for col in labs_to_keep for stat in lab_stats]
df_win = df_win.sort_values(['feat_idx', 'minutes_since_intime'])
df_win['tv'] = df_win['minutes_since_intime'] * df_win['valuenum']
df_win['tt'] = df_win['minutes_since_intime'] ** 2
grp = df_win.groupby(['feat_idx', 'label'])
agg = grp.agg(
    mean=('valuenum', 'mean'), median=('valuenum', 'median'),
    min=('valuenum', 'min'), max=('valuenum', 'max'),
    first=('valuenum', 'first'), last=('valuenum', 'last'), n=('valuenum', 'size'),
    sum_t=('minutes_since_intime', 'sum'), sum_v=('valuenum', 'sum'),
    sum_tv=('tv', 'sum'), sum_tt=('tt', 'sum'),
)
agg['std'] = grp['valuenum'].std(ddof=0)
n = agg['n']
cov = agg['sum_tv'] / n - (agg['sum_t'] / n) * (agg['sum_v'] / n)
vart = agg['sum_tt'] / n - (agg['sum_t'] / n) ** 2
agg['coeff'] = np.where((n >= 2) & (vart > 0), cov / vart, 0.0)
agg['rate_change'] = np.where(agg['first'] != 0, (agg['last'] - agg['first']) / agg['first'], np.nan)

# Arrange laboratory summaries into one row per stay
wide = agg[lab_stats].unstack('label')
wide.columns = [f"labvalues_{label}_{stat}" for stat, label in wide.columns]
lab_features_df = wide.reindex(index=np.arange(len(df_features)), columns=full_cols)
lab_features_df.index = df_features.index

df_features = pd.concat([df_features, lab_features_df], axis=1)
df_features = finalise_features(df_features, lbls_diags, decompensation_model_cols)
df_features.to_csv(output_dir / 'mds_ed_static.csv', index=False)
print(f"Done. Wrote {output_dir / 'mds_ed_static.csv'} ({len(df_features)} rows, {len(df_features.columns)} cols, {df_features['general_study_id'].nunique()} unique patients, {df_features['general_ed_stay_id'].nunique()} unique stays)", flush=True)
print(f"Total time: {datetime.timedelta(seconds=round(time.time() - _start_time))}", flush=True)
