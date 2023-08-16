import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/src/")


from pythae.ssc.cohort import Cohort
from pythae.ssc.lung import LUNG_ILD
from pythae.ssc.heart import HEART
import pickle

if __name__ == "__main__":
    local = True
    if local:
        data_path = "/home/cctrotte/krauthammer/eustar/fake_data/raw/"
        save_path = "/home/cctrotte/krauthammer/eustar/fake_data/processed/"
    else:
        data_path = "/cluster/work/medinfmk/EUSTAR2/data/raw/"
        save_path = "/cluster/work/medinfmk/EUSTAR2/data/processed/ct/"
    cohort = Cohort(data_path)
    cohort.preprocess(ns_visits_drop=[1, 2, 3, 4] + [i for i in range(15, 35)])
    cut_time = padd_time = 15
    Patients = cohort.create_patients()
    Patients_train, Patients_test = cohort.split_train_test()
    organs = {"lung": LUNG_ILD(), "heart": HEART()}
    name = "_t"
    for key, organ in organs.items():
        cohort.extract_data_frame_for_organ(organ)
        cohort.create_labels_for_organ(organ)
        cohort.encode_data_for_organ(organ)
        # dat_train, dat_test = cohort.data_train_test_for_organ(organ, save_path, name=name + organ.name)

    with open(save_path + "organs.pkl", "wb") as file:
        pickle.dump(organs, file)
    # save also the cohort
    with open(save_path + "cohort" + name + ".pkl", "wb") as file:
        pickle.dump(cohort, file)
