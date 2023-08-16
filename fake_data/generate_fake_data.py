import pandas as pd
import string
import random
import time
import datetime
import numpy as np
import pickle
import sys

sys.path.append("/home/cctrotte/krauthammer/eustar/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ct/")
from fake_data.utils import *

sys.path.append("/home/cctrotte/krauthammer/eustar/benchmark_VAE/src/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ct/benchmark_VAE/src/")


params = {
    "patients": {
        "n_patients": 2000,
        "min_num_visits": 5,
        "max_num_visits": 30,
        "features": {"Sex": {"values": ["female", "male"], "kind": "categorical", "nan_prop": 0.0}, "Height": {"values": [60, 220], "kind": "continuous", "nan_prop": 0.2},
                     "Race white": {"values": ["White"], "kind": "categorical", "nan_prop": 0.2}, "Hispanic": {"values": ["Hispanic"], "kind": "categorical", "nan_prop": 0.7},
                     "Any other white": {"values": ["Any other White"], "kind": "categorical", "nan_prop": 0.7}, 
                     "Race asian": {"values": ["Asian"], "kind": "categorical", "nan_prop": 0.9},
                     "Race black": {"values": ["Black"], "kind": "categorical", "nan_prop": 0.9},
                     "Subsets of SSc according to LeRoy (1988)": {"values": ["Limited cutaneous SSc", "Diffuse cutaneous SSc"], "kind": "categorical", "nan_prop": 0.8}}
    },
    "visits": {
        "features": {
            "Forced Vital Capacity (FVC - % predicted)": {
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.9,
            },
            "DLCO/SB (% predicted)": {
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.9,
            },
            "DLCOc/VA (% predicted)": {
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.9,
            },
            "Lung fibrosis/ %involvement": {
                "values": ["Unknown", ">20%", "<20%", "Indeterminate"],
                "kind": "categorical",
                "nan_prop": 0.9,
            },
            "Dyspnea (NYHA-stage)": {
                "values": ["1", "2", "3", "4", "Unknown"],
                "kind": "categorical",
                "nan_prop": 0.9,
            },
            "Worsening of cardiopulmonary manifestations within the last month": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.9,
            },
            "HRCT: Lung fibrosis": {
                "values": ["No", "Unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.9,
            },
            "Ground glass opacification": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.95,
            },
            "Honey combing": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.4,
            },
            "Tractions": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.5,
            },
            "Any reticular changes": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.4,
            },
            "BNP (pg/ml)": {"values": [0, 3000], "kind": "continuous", "nan_prop": 0.5},
            "Left ventricular ejection fraction (%)": {
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.3,
            },
            "Worsening of cardiopulmonary manifestations within the last month": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "Diastolic function abnormal": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "Ventricular arrhythmias": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "Arrhythmias requiring therapy": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "Pericardial effusion on echo": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "Conduction blocks": {
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            "NTproBNP (pg/ml)": {
                "values": [0, 1000],
                "kind": "continuous",
                "nan_prop": 0.5,
            },
            "PAPsys (mmHg)":{
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            "TAPSE: tricuspid annular plane systolic excursion in cm":{
                "values": [0, 40],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            'Right ventricular area (cm2) (right ventricular dilation)':{
                "values": [0, 60],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            "Tricuspid regurgitation velocity (m/sec)":{
                "values": [0, 10],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            'Pulmonary wedge pressure (mmHg)':{
                "values": [0, 100],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            'Pulmonary resistance (dyn.s.cm-5)':{
                "values": [0, 1000],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            '6 Minute walk test (distance in m)':{
                "values": [0, 1000],
                "kind": "continuous",
                "nan_prop": 0.8,
            },
            'Auricular Arrhythmias':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'BNP (pg/ml)':{
                "values": [0, 2500],
                "kind": "continuous",
                "nan_prop": 0.6,
            },
            'Cardiac arrhythmias':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Renal crisis':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Worsening of skin within the last month':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Extent of skin involvement':{
                "values": ['Limited cutaneous involvement', 'Diffuse cutaneous involvement',
                'Only sclerodactyly', 'No skin involvement'],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Modified Rodnan Skin Score, only imported value':{
                "values": [0, 20],
                "kind": "continuous",
                "nan_prop": 0.6,
            },
            'Skin thickening of the fingers of both hands extending proximal to the MCP joints':{
                "values": ['Never', 'Current', 'Previously'],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Skin thickening of the whole finger distal to MCP (Sclerodactyly)':{
                "values": ['Never', 'Current', 'Previously'],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Skin thickening sparing the fingers':{
                "values": ['Never', 'Current', 'Previously'],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Joint synovitis':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Tendon friction rubs':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Joint polyarthritis':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Swollen joints':{
                "values": [0,40],
                "kind": "continuous",
                "nan_prop": 0.3,
            },
            'DAS 28 (ESR, calculated)':{
                "values": [0,10],
                "kind": "continuous",
                "nan_prop": 0.3,
            },
            'DAS 28 (CRP, calculated)':{
                "values": [0,10],
                "kind": "continuous",
                "nan_prop": 0.3,
            },
            'Muscle weakness':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Proximal muscle weakness not explainable by other causes':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Muscle atrophy':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Myalgia':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Stomach symptoms (early satiety, vomiting)':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Intestinal symptoms (diarrhea, bloating, constipation)':{
                "values": ["No", "unknown", "Yes"],
                "kind": "categorical",
                "nan_prop": 0.3,
            },
            'Body weight (kg)':{
                "values": [0,200],
                "kind": "continuous",
                "nan_prop": 0.2,
            },
        }
    },
    "meds": {},
}

if __name__ == "__main__":
    n_patients = params["patients"]["n_patients"]
    patient_ids = get_patient_ids(n_patients)
    pats = pd.DataFrame({"Id Patient V2018": patient_ids})
    vis = pd.DataFrame()
    num_visit_per_patient = random.choices(
        range(
            params["patients"]["min_num_visits"],
            params["patients"]["max_num_visits"] + 1,
        ),
        weights=None,
        k=n_patients,
    )
    vis_patient_ids = []
    for i in range(len(patient_ids)):
        vis_patient_ids += [patient_ids[i]] * num_visit_per_patient[i]
    vis["Id Patient V2018"] = vis_patient_ids
    for key, value in params["visits"]["features"].items():
        if value["kind"] == "continuous":
            values_complete = np.random.uniform(
                value["values"][0], value["values"][1], size=len(vis)
            )
            vis[key] = insert_nans(values_complete, value["nan_prop"])
        elif value["kind"] == "categorical":
            values_complete = random.choices(value["values"], weights=None, k=len(vis))
            vis[key] = insert_nans(values_complete, value["nan_prop"])
    for key, value in params["patients"]["features"].items():
        if value["kind"] == "continuous":
            values_complete = np.random.uniform(
                value["values"][0], value["values"][1], size=len(pats)
            )
            pats[key] = insert_nans(values_complete, value["nan_prop"])
        elif value["kind"] == "categorical":
            values_complete = random.choices(value["values"], weights=None, k=len(pats))
            pats[key] = insert_nans(values_complete, value["nan_prop"])
    vis["Visit Date"] = [random_date(start="1980-01-01") for i in range(len(vis))]
    pats["Date of birth"] = [random_date(start="1920-01-01") for i in range(len(pats))]
    pats["Onset of first non-Raynaud?s of the disease"] = [random_date(start="1980-01-01") for i in range(len(pats))]
    meds = pd.DataFrame({"Id Patient 2018": patient_ids})
    path = "/home/cctrotte/krauthammer/eustar/fake_data/raw/"
    for df, name in zip([pats, vis, meds], ["pats", "vis", "meds"]):
        with open(path + name, "wb") as file:
            pickle.dump(df, file)

    print(f"Fake data generated! and stored in {path}")
