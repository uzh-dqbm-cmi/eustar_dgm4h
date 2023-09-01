import sys
import pickle

sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/src/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ml4h_ct/benchmark_VAE/src/")
import warnings
import copy
from pythae.ssc.body import Body

warnings.filterwarnings("ignore")
from pythae.ssc.organs.lung import LUNG_ILD
from pythae.ssc.organs.heart import HEART
from pythae.ssc.organs.lung_ph import LUNG_PH
from pythae.ssc.organs.kidney import KIDNEY
from pythae.ssc.organs.skin import SKIN
from pythae.ssc.organs.arthritis import ARTHRITIS
from pythae.ssc.organs.muscle import MUSCLE
from pythae.ssc.organs.gastro import GASTRO
from pythae.ssc.organs.general import GENERAL
from pythae.ssc.organs.static import STATIC

from pythae.ssc.cohort import Cohort
import numpy as np
from pythae.ssc.utils import compute_folds, save_cv


class CV:
    def __init__(self, organs, data_path, save_path, n_folds=5):
        self.n_folds = n_folds
        self.bodies = [Body(organs) for i in range(n_folds)]
        cohort = Cohort(data_path)
        cohort.preprocess(ns_visits_drop=[1, 2, 3, 4] + [i for i in range(15, 35)])
        Patients = cohort.create_patients()
        self.cohorts = compute_folds(
            cohort, Patients, n_folds=n_folds, frac_train=0.85, seed=0
        )
        for cohort, body in zip(self.cohorts, self.bodies):
            cohort.extract_data_frame(body)
            for organ in body.organs:
                cohort.create_labels_for_organ(organ, use_body_df=True)
            cohort.encode_data(body)
        save_cv(self.cohorts, self.bodies[0], save_path, name="_ml4h", PICKLE=True)


if __name__ == "__main__":
    local = True
    name = "_ml4h"
    # change for 5 fold cv
    n_folds = 2 if local else 5
    
    if local:
        data_path = "/home/cctrotte/krauthammer/eustar_clean/fake_data/raw/"
        save_path = "/home/cctrotte/krauthammer/eustar_clean/fake_data/processed/"
    else:
        data_path = "/cluster/work/medinfmk/EUSTAR2/data/raw/"
        save_path = "/cluster/work/medinfmk/EUSTAR2/data/processed/ct/"
    # medium dataset
    organs = [
        LUNG_ILD(),
        HEART(),
        LUNG_PH(),
        STATIC(),
        ARTHRITIS(),
    ]

    cv = CV(organs, data_path, save_path, n_folds=n_folds)

    with open(save_path + "bodies_" + name + ".pkl", "wb") as file:
        pickle.dump(cv.bodies, file)
    # save also the cohort
    with open(save_path + "cohorts_" + name + ".pkl", "wb") as file:
        pickle.dump(cv.cohorts, file)

    print("end")
