import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random

def create_random_tensor(n):
    tensor = torch.zeros(n)  # Initialize tensor with zeros
    random_index = random.randint(0, n - 1)  # Generate a random index within the range of n
    tensor[random_index] = 1  # Set the value at the random index to 1
    return tensor
def create_one_tensor(n):
    tensor = torch.zeros(n)  # Initialize tensor with zeros
    tensor[0] = 1  # Set the value index to 1
    return tensor

def fill_tensor(data, mask, cat=False):
    filled_data = torch.zeros_like(data)  # Initialize the filled tensor with zeros
    if cat:
         filled_data[0] = create_one_tensor(data.shape[1])
    else:
        filled_data[0] = data[0]  # Copy the first row of data as it is

    # Iterate over columns and rows starting from the second row
    for i in range(1, data.size(0)):
        filled_data[i] = torch.where(mask[i] == 1, data[i], filled_data[i - 1])

    return filled_data
def get_result_df(model, body, splits_x0, names_x0, kinds_x0, out2, data_x, data_x_recon, data_x_splitted, non_missing_x, non_missing_x_splitted, non_missing_x_recon, delta_t_resc, num_rec_for_pred ):
    if model.sample_z:
        res_matrix, probs_matrix, res_list = body.decode_preds(out2.recon_x, splits_x0, names_x0)
        res_list_samples = body.decode_preds(out2.recon_x, splits_x0, names_x0)[2]
    else:
        res_matrix, probs_matrix, res_list = body.decode_preds(out2.recon_m, splits_x0, names_x0)
        res_list_samples = body.decode_preds(out2.recon_m, splits_x0, names_x0)[2]
    ground_truth = body.decode(data_x_recon, splits_x0, names_x0)
    # def count_within_range(x_true, x_pred, x_var_pred):
    #     lower_bound = x_pred - 2 * torch.sqrt(x_var_pred)
    #     upper_bound = x_pred + 2 * torch.sqrt(x_var_pred)
    #     within_range = (x_true >= lower_bound) & (x_true <= upper_bound)
    #     count = torch.sum(within_range).item()

    #     return count
    list_ = np.concatenate(([0], np.cumsum(splits_x0)))
    patient_specific_baseline = []
    for index, elem in enumerate(list_[:-1]):
        naive_all = []
        if kinds_x0[index] == 'continuous':
            for pat in range(len(data_x_splitted)):
                new_naive = data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                mean_cohort = torch.mean(data_x[non_missing_x[:, index]>0, index])
                mask = (non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0)
                new_naive[~mask] = mean_cohort
                #new_naive = torch.cat([torch.full(new_naive.shape, mean_cohort),torch.full(new_naive.shape, mean_cohort), torch.cat([new_naive[index].repeat(new_naive.shape) for index in range(len(new_naive)-1)])])
                new_naive = torch.cat([torch.full(new_naive.shape, mean_cohort),torch.full(new_naive.shape, mean_cohort), torch.cat([torch.cat([torch.tensor([[mean_cohort]]), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                naive_all.append(new_naive)
        else:
            for pat in range(len(data_x_splitted)):
                new_naive = data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                mean_cohort = torch.mean(data_x[non_missing_x[:, index]>0, index])
                mask = (non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0).any(dim = 1)
                new_naive[~mask] = create_one_tensor(new_naive.shape[1])
                #new_naive = torch.cat([create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), torch.cat([new_naive[index].repeat(len(new_naive), 1) for index in range(len(new_naive)-1)])])
                new_naive = torch.cat([create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), torch.cat([torch.cat([create_one_tensor(new_naive.shape[1]).reshape(1,-1), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                naive_all.append(new_naive)
        patient_specific_baseline.append(torch.cat(naive_all))
    patient_specific_baseline = torch.cat(patient_specific_baseline, dim = 1)
    patient_specific_baseline_ff = []
    for index, elem in enumerate(list_[:-1]):
        naive_all = []
        if kinds_x0[index] == 'continuous':
            for pat in range(len(data_x_splitted)):
                new_naive = data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                mean_cohort = torch.mean(data_x[non_missing_x[:, index]>0, index])
                mask = (non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0)
                new_naive = fill_tensor(new_naive, mask)
                new_naive = torch.cat([torch.full(new_naive.shape, mean_cohort),torch.full(new_naive.shape, mean_cohort), torch.cat([torch.cat([torch.tensor([[mean_cohort]]), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                naive_all.append(new_naive)
        else:
            for pat in range(len(data_x_splitted)):
                new_naive = data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                mean_cohort = torch.mean(data_x[non_missing_x[:, index]>0, index])
                mask = (non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0).any(dim = 1).reshape(-1, 1).repeat(new_naive.shape)
                new_naive = fill_tensor(new_naive, mask, cat = True)
                new_naive = torch.cat([create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), torch.cat([torch.cat([create_one_tensor(new_naive.shape[1]).reshape(1,-1), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                naive_all.append(new_naive)

        patient_specific_baseline_ff.append(torch.cat(naive_all))
    patient_specific_baseline_ff = torch.cat(patient_specific_baseline_ff, dim = 1)
    cat_baseline_ff = body.decode(patient_specific_baseline_ff, splits_x0, names_x0)
    cat_baseline = body.decode(patient_specific_baseline, splits_x0, names_x0)
    time_flags = [(0,0), (0, 1), (1, 2), (2, 3), (3, 4),  (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12)]
    #time_flags = [(-7,-6), (-6, -5), (-5,-4), (-4,-3), (-3,-2), (-2,-1), (-1, 0), (0,0)]

    #for i, sample in enumerate(samples):
    dfs_cont = {name: pd.DataFrame(columns = ["count", "mae", "mae_naive", "mae_pat_spec", "mae_pat_spec_ff"], index = time_flags) for i, name in enumerate(names_x0) if kinds_x0[i] == "continuous"}
    dfs_cat = {name: pd.DataFrame(columns = ["acc", "naive acc", "pat_spec_ff", "pat_spec"], index = time_flags) for i, name in enumerate(names_x0) if kinds_x0[i] != "continuous"}
    dfs_cont_scaled = {name: pd.DataFrame(columns = ["count", "mae", "mae_naive", "mae_pat_spec", "mae_pat_spec_ff"], index = time_flags) for i, name in enumerate(names_x0) if kinds_x0[i] == "continuous"}
    for j, interv in enumerate(time_flags):
        if interv[0] == interv[1]:
            to_keep = torch.tensor((delta_t_resc == interv[0]) & (num_rec_for_pred > 0)).flatten()
        else:
            to_keep = torch.tensor((delta_t_resc > interv[0]) & (delta_t_resc <= interv[1])  & (num_rec_for_pred > 0)).flatten()
        print(to_keep.count_nonzero().item())
        list_ = np.concatenate(([0], np.cumsum(splits_x0)))
        for index, elem in enumerate(list_[:-1]):
            name = names_x0[index]
            print(name)
            if kinds_x0[index] == 'continuous':
                data_sc = data_x_recon[(non_missing_x_recon[:, elem:list_[index+1]] > 0).flatten(), elem:list_[index+1]]
                all_targets = body.get_var_by_name(name).decode(data_sc)
                print(all_targets.mean())
                mask_ = non_missing_x_recon[to_keep,elem:list_[index+1]] > 0
                recon = res_list_samples[index][0][to_keep][mask_].detach()
                true = data_x_recon[to_keep, elem:list_[index+1]][mask_].detach()
                pat_baseline = patient_specific_baseline[to_keep, elem:list_[index+1]][mask_].detach()
                pat_baseline_ff = patient_specific_baseline_ff[to_keep, elem:list_[index+1]][mask_].detach()

                if len(recon) > 0:
                    recon_resc = body.get_var_by_name(name).decode(recon.reshape(-1, 1))
                    true_resc = body.get_var_by_name(name).decode(true.reshape(-1, 1))
                    patient_specific_baseline_resc = body.get_var_by_name(name).decode(pat_baseline.reshape(-1, 1))
                    patient_specific_baseline_ff_resc = body.get_var_by_name(name).decode(pat_baseline_ff.reshape(-1, 1))
                    mse = sum((recon_resc- true_resc)**2)/len(recon)
                    mae = sum(abs(recon_resc- true_resc))/len(recon)
                    mae_naive = sum(abs(np.mean(all_targets) - true_resc))/len(true_resc)
                    mae_pat_spec = sum(abs(patient_specific_baseline_resc - true_resc))/len(true_resc)
                    mae_pat_spec_ff = sum(abs(patient_specific_baseline_ff_resc - true_resc))/len(true_resc)
                    mse_sc = sum((recon- true)**2)/len(recon)
                    mae_sc = sum(abs(recon- true))/len(recon)
                    mae_naive_sc = sum(abs(np.mean(np.array(data_sc)) - true))/len(true)
                    mae_pat_spec_sc = sum(abs(pat_baseline - true))/len(true)
                    mae_pat_spec_ff_sc = sum(abs(pat_baseline_ff - true))/len(true)

                    dfs_cont[names_x0[index]].iloc[j]["count"] = len(recon) 
                    dfs_cont[names_x0[index]].iloc[j]["mae"] = mae.item()
                    dfs_cont[names_x0[index]].iloc[j]["mae_naive"] = mae_naive.item()
                    dfs_cont[names_x0[index]].iloc[j]["mae_pat_spec"] = mae_pat_spec.item()
                    dfs_cont[names_x0[index]].iloc[j]["mae_pat_spec_ff"] = mae_pat_spec_ff.item()
                    dfs_cont_scaled[names_x0[index]].iloc[j]["count"] = len(recon) 
                    dfs_cont_scaled[names_x0[index]].iloc[j]["mae"] = mae_sc.item()
                    dfs_cont_scaled[names_x0[index]].iloc[j]["mae_naive"] = mae_naive_sc.item()
                    dfs_cont_scaled[names_x0[index]].iloc[j]["mae_pat_spec"] = mae_pat_spec_sc.item()
                    dfs_cont_scaled[names_x0[index]].iloc[j]["mae_pat_spec_ff"] = mae_pat_spec_ff_sc.item()
                else:
                    dfs_cont[names_x0[index]].iloc[j]["count"] = 0
                    dfs_cont[names_x0[index]].iloc[j]["mae"] = np.nan
                    dfs_cont[names_x0[index]].iloc[j]["mae_naive"] = np.nan
                    dfs_cont[names_x0[index]].iloc[j]["mae_pat_spec"] = np.nan
                    dfs_cont[names_x0[index]].iloc[j]["mae_pat_spec_ff"] = np.nan


            elif kinds_x0[index] != 'continuous':
                all_targets = ground_truth[1][index][(non_missing_x_recon[:, elem:list_[index+1]] > 0).any(dim =1)]
                mask_ = (non_missing_x_recon[to_keep,elem:list_[index+1]] > 0).any(dim =1)
                recon =  res_list_samples[index][0][to_keep]
                true = ground_truth[1][index][to_keep][mask_]
                pat_spec_ff = cat_baseline_ff[1][index][to_keep][mask_]
                pat_spec = cat_baseline[1][index][to_keep][mask_]

                recon = body.get_var_by_name(names_x0[index]).get_categories(recon)[mask_]
                if len(recon) > 0:
                    acc = accuracy_score(true.flatten().astype(float), recon)
                    value, counts = np.unique(all_targets.flatten(), return_counts = True)
                    naive = np.random.choice(value, size = len(true.flatten()), p = counts/sum(counts))
                    naive_acc = accuracy_score(naive.astype(float), true.flatten().astype(float))
                    pat_spec_acc_ff = accuracy_score(pat_spec_ff, true.flatten().astype(float))
                    pat_spec_acc = accuracy_score(pat_spec, true.flatten().astype(float))

                    print(f'acc {acc}')
                    print(f'acc naive {naive_acc}')
                    #print(f'{classification_report(true.flatten().astype(float), recon)}')
                    dfs_cat[names_x0[index]].iloc[j]["acc"] = acc
                    dfs_cat[names_x0[index]].iloc[j]["naive acc"] = naive_acc
                    dfs_cat[names_x0[index]].iloc[j]["pat_spec_ff"] = pat_spec_acc_ff
                    dfs_cat[names_x0[index]].iloc[j]["pat_spec"] = pat_spec_acc
                else:
                    dfs_cat[names_x0[index]].iloc[j]["count"] = 0
                    dfs_cat[names_x0[index]].iloc[j]["mae"] = np.nan
                    dfs_cat[names_x0[index]].iloc[j]["mae_naive"] = np.nan
                    dfs_cat[names_x0[index]].iloc[j]["mae_pat_spec"] = np.nan
                    dfs_cat[names_x0[index]].iloc[j]["mae_pat_spec_ff"] = np.nan
    tmp = pd.DataFrame(np.nanmean(np.array([elem for elem in dfs_cont_scaled.values()], dtype = np.float64), axis =0), index = dfs_cont_scaled['Forced Vital Capacity (FVC - % predicted)'].index, columns = dfs_cont_scaled['Forced Vital Capacity (FVC - % predicted)'].columns)
    
    return tmp

class Evaluation:
    def __init__(self, data_test, model, body, splits_x0, names_x0, kinds_x0, splits_y0, names_y0, kinds_y0, size):
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = model
        self.body = body
        self.splits_x0 = splits_x0
        self.names_x0 = names_x0
        self.kinds_x0 = kinds_x0
        self.splits_y0 = splits_y0
        self.names_y0 = names_y0
        self.kinds_y0 = kinds_y0
        self.batch = data_test.get_ith_sample_batch_with_customDataLoader(0,size)
        self.data_x = self.batch["data_x"]
        self.data_s = self.batch["data_s"]
        self.non_missing_x = 1 - self.batch["missing_x"] * 1.0
        self.splits = self.batch["splits"]  # N_patients
        self.times = self.batch["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1
        self.non_missing_y = 1 - self.batch["missing_y"] * 1.0  # N_patients x n_class
        self.data_y = self.batch["data_y"]  # N_patients x n_class
        self.data_x_splitted = torch.split(self.data_x, self.splits, dim=0)
        self.data_y_splitted = torch.split(self.data_y, self.splits, dim=0)
        self.data_s_splitted = torch.split(self.data_s, self.splits, dim=0)
        self.non_missing_x_splitted = torch.split(self.non_missing_x, self.splits, dim=0)
        self.non_missing_y_splitted = torch.split(self.non_missing_y, self.splits, dim=0)
        self.times_splitted = torch.split(self.times, self.splits, dim=0)
        self.data_x_recon = torch.cat([self.data_x_splitted[pat].repeat(self.splits[pat] + 1,1) for pat in range(len(self.splits))])
        self.data_s_recon = torch.cat([self.data_s_splitted[pat].repeat(self.splits[pat]+ 1,1) for pat in range(len(self.splits))])
        self.data_y_recon = torch.cat([self.data_y_splitted[pat].repeat(self.splits[pat] + 1,1) for pat in range(len(self.splits))])
        self.non_missing_x_recon = torch.cat([self.non_missing_x_splitted[pat].repeat(self.splits[pat]+1,1) for pat in range(len(self.splits))])
        self.non_missing_y_recon = torch.cat([self.non_missing_y_splitted[pat].repeat(self.splits[pat]+1,1) for pat in range(len(self.splits))])
        self.times_recon = torch.cat([self.times_splitted[pat].repeat(self.splits[pat]+1,1) for pat in range(len(self.splits))])
        self.indices_recon = torch.cat([torch.cat([torch.cat([torch.full((index, 1), True), torch.full((self.splits[pat] - index,1), False)], dim=0) for index in range(0, self.splits[pat] + 1)]) for pat in range(len(self.splits))]).flatten()
        self.num_rec_for_pred = np.array(torch.cat([torch.cat([torch.full((self.splits[pat], 1), index) for index in range(0, self.splits[pat] + 1)]) for pat in range(len(self.splits))]))
        if not self.model.model_config.retrodiction:
            self.absolute_times = torch.cat([torch.cat([self.times_splitted[pat][0].repeat(len(self.times_splitted[pat]), 1),                    
                                                               torch.cat(
                                    [
                                        torch.cat(
                                            [
                                                self.times_splitted[pat][:index, :],
                                                self.times_splitted[pat][index, :].repeat(
                                                    len(self.times_splitted[pat]) - index, 1
                                                ),
                                            ],
                                            dim=0,
                                        )
                                        for index in range(len(self.times_splitted[pat]))
                                    ]
                                )]) for pat in range(len(self.times_splitted))])
        else:
            self.absolute_times = torch.cat([torch.cat([self.times_splitted[pat][0].repeat(len(self.times_splitted[pat]), 1), torch.cat([


                    elem.repeat(
                        len(self.times_splitted[pat]), 1
                    ) for elem in self.times_splitted[pat]])]) for pat in range(len(self.times_splitted))])
        self.absolute_times_resc = self.body.get_var_by_name('time [years]').decode(self.absolute_times)
        self.times_recon_resc = self.body.get_var_by_name('time [years]').decode(self.times_recon)  
        self.delta_t_resc = self.times_recon_resc - self.absolute_times_resc
     
    def evaluate(self, num_samples = 10, evaluate_y = True):
        self.predictions = self.model(self.batch)
        self.samples = [self.model(self.batch) for index in range(num_samples)] 
        self.ground_truth_x = self.body.decode(self.data_x_recon, self.splits_x0, self.names_x0)
        if self.model.sample_z:
            self.res_matrix, self.probs_matrix, self.res_list = self.body.decode_preds(self.predictions.recon_x, self.splits_x0, self.names_x0)
            self.res_list_samples = [self.body.decode_preds(sample.recon_x, self.splits_x0, self.names_x0)[2] for sample in self.samples]
        else:
            self.res_matrix, self.probs_matrix, self.res_list = body.decode_preds(self.predictions.recon_m, self.splits_x0, self.names_x0)
            self.res_list_samples = [self.body.decode_preds(sample.recon_m, self.splits_x0, self.names_x0)[2] for sample in self.samples]
        if evaluate_y:
            self.ground_truth_y = self.body.decode(self.data_y_recon, self.splits_y0, self.names_y0)
            self.res_matrix_y, self.probs_matrix_y, self.res_list_y = self.body.decode_preds(self.predictions.y_out_rec, self.splits_y0, self.names_y0)
            self.predicted_cats_y = torch.empty_like(self.res_matrix_y)

            for index, var in enumerate(self.names_y0): 
                self.predicted_cats_y[:,index] = self.body.get_var_by_name(var).get_categories(self.res_matrix_y[:,index])

    def get_patient_specific_baseline_x(self):

        list_ = np.concatenate(([0], np.cumsum(splits_x0)))
        patient_specific_baseline = []
        # iterate over x variables
        for index, elem in enumerate(list_[:-1]):
            naive_all = []
            if self.kinds_x0[index] == 'continuous':
                for pat in range(len(self.data_x_splitted)):
                    # copy patient data
                    new_naive = self.data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                    # store mean value of cohort for imputation of missing values
                    mean_cohort = torch.mean(self.data_x[self.non_missing_x[:, index]>0, index])
                    # available values
                    mask = (self.non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0)
                    # fill missing values with previous value of patient
                    new_naive = self.fill_tensor(new_naive, mask)
                    # shift predictions so that we always predict the last available value.
                    # first concatenate two tensors (ie predictions before any info is available) filled with the mean value of the cohort
                    # then for m=1, ... T -1, recursively fill tensor so that if m values are available for prediction, we predict [mean_cohort, data[1], .., data[m], data[m], ..., data[m]] for ground truth [data[1], ..., data[T]] 
                    new_naive = torch.cat([torch.full(new_naive.shape, mean_cohort),torch.full(new_naive.shape, mean_cohort), torch.cat([torch.cat([torch.tensor([[mean_cohort]]), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                    naive_all.append(new_naive)
            else:
                for pat in range(len(self.data_x_splitted)):
                    new_naive = self.data_x_splitted[pat][: ,elem:list_[index+1]].clone()
                    #mean_cohort = torch.mean(self.data_x[non_missing_x[:, index]>0, index])
                    mask = (self.non_missing_x_splitted[pat][:,  elem:list_[index+1]] > 0).any(dim = 1).reshape(-1, 1).repeat(new_naive.shape)
                    new_naive = self.fill_tensor(new_naive, mask, cat = True)
                    # same as for continuous, but instead of filling with mean of the cohort we fill with first value
                    new_naive = torch.cat([self.create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), self.create_one_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), torch.cat([torch.cat([self.create_one_tensor(new_naive.shape[1]).reshape(1,-1), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                    naive_all.append(new_naive)

            patient_specific_baseline.append(torch.cat(naive_all))
        self.patient_specific_baseline_x = torch.cat(patient_specific_baseline, dim = 1)
        self.cat_baseline_x = self.body.decode(self.patient_specific_baseline_x, self.splits_x0, self.names_x0)

    def get_patient_specific_baseline_y(self):
        patient_specific_baseline_y = []
        list_y = np.concatenate(([0], np.cumsum(self.splits_y0)))

        for index, elem in enumerate(list_y[:-1]):
            naive_all = []
            if self.kinds_y0[index] == 'continuous':
                print(index)
                for pat in range(len(self.data_y_splitted)):
                    new_naive = self.data_y_splitted[pat][: ,elem:list_y[index+1]].clone()
                    mean_cohort = torch.mean(self.data_y[self.non_missing_y[:, index]>0, index])
                    mask = (self.non_missing_y_splitted[pat][:,  elem:list_y[index+1]] > 0)
                    new_naive = fill_tensor(new_naive, mask)
                    new_naive = torch.cat([torch.full(new_naive.shape, mean_cohort),torch.full(new_naive.shape, mean_cohort), torch.cat([torch.cat([torch.tensor([[mean_cohort]]), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                    naive_all.append(new_naive)
            else:
                for pat in range(len(self.data_y_splitted)):
                    new_naive = self.data_y_splitted[pat][: ,elem:list_y[index+1]].clone()
                    mean_cohort = torch.mean(self.data_y[self.non_missing_y[:, index]>0, index])
                    mask = (self.non_missing_y_splitted[pat][:,  elem:list_y[index+1]] > 0).any(dim = 1).reshape(-1, 1).repeat(new_naive.shape)
                    new_naive = self.fill_tensor(new_naive, mask, cat = True)
                    new_naive = torch.cat([self.create_random_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), self.create_random_tensor(new_naive.shape[1]).repeat(len(new_naive), 1), torch.cat([torch.cat([self.create_random_tensor(new_naive.shape[1]).reshape(1,-1), new_naive[:index], new_naive[index].repeat(len(new_naive) - index -1, 1)], dim = 0) for index in range(len(new_naive)-1)])])
                    naive_all.append(new_naive)

            patient_specific_baseline_y.append(torch.cat(naive_all))
        self.patient_specific_baseline_y = torch.cat(patient_specific_baseline_y, dim = 1)
        self.cat_baseline_y,_ = self.body.decode(self.patient_specific_baseline_y, self.splits_y0, self.names_y0)

    def get_result_df_x(self, time_flags = [(0,0), (0, 1), (1, 2), (2, 3), (3, 4),  (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12)]):
        dfs_cont = {name: pd.DataFrame(columns = ["count", "mae", "mae_naive", "mae_pat_spec"], index = time_flags) for i, name in enumerate(self.names_x0) if self.kinds_x0[i] == "continuous"}
        dfs_cat = {name: pd.DataFrame(columns = ["acc", "naive acc", "pat_spec"], index = time_flags) for i, name in enumerate(self.names_x0) if self.kinds_x0[i] != "continuous"}
        dfs_cat_f1 = {name: pd.DataFrame(columns = ["f1", "naive f1",  "pat_spec"], index = time_flags) for i, name in enumerate(self.names_x0) if self.kinds_x0[i] != "continuous"}
        dfs_cont_scaled = {name: pd.DataFrame(columns = ["count", "mae", "mae_naive", "mae_pat_spec" ], index = time_flags) for i, name in enumerate(self.names_x0) if self.kinds_x0[i] == "continuous"}
        for j, interv in enumerate(time_flags):
            if interv[0] == interv[1]:
                to_keep = torch.tensor((self.delta_t_resc == interv[0]) & (self.num_rec_for_pred > 0)).flatten()
            else:
                to_keep = torch.tensor((self.delta_t_resc > interv[0]) & (self.delta_t_resc <= interv[1])  & (self.num_rec_for_pred > 0)).flatten()
            print(to_keep.count_nonzero().item())
            list_ = np.concatenate(([0], np.cumsum(self.splits_x0)))
            for index, elem in enumerate(list_[:-1]):
                name = self.names_x0[index]
                print(name)
                if self.kinds_x0[index] == 'continuous':
                    data_sc = self.data_x_recon[(self.non_missing_x_recon[:, elem:list_[index+1]] > 0).flatten(), elem:list_[index+1]]
                    all_targets = self.body.get_var_by_name(name).decode(data_sc)
                    mask_ = self.non_missing_x_recon[to_keep,elem:list_[index+1]] > 0
                    recon = self.res_list[index][0][to_keep][mask_].detach()
                    true = self.data_x_recon[to_keep, elem:list_[index+1]][mask_].detach()
                    pat_baseline_ff = self.patient_specific_baseline_x[to_keep, elem:list_[index+1]][mask_].detach()

                    if len(recon) > 0:
                        recon_resc = self.body.get_var_by_name(name).decode(recon.reshape(-1, 1))
                        true_resc = self.body.get_var_by_name(name).decode(true.reshape(-1, 1))
                        patient_specific_baseline_ff_resc = self.body.get_var_by_name(name).decode(pat_baseline_ff.reshape(-1, 1))
                        mse = sum((recon_resc- true_resc)**2)/len(recon)
                        mae = sum(abs(recon_resc- true_resc))/len(recon)
                        mae_naive = sum(abs(np.mean(all_targets) - true_resc))/len(true_resc)
                        mae_pat_spec_ff = sum(abs(patient_specific_baseline_ff_resc - true_resc))/len(true_resc)
                        mse_sc = sum((recon- true)**2)/len(recon)
                        mae_sc = sum(abs(recon- true))/len(recon)
                        mae_naive_sc = sum(abs(np.mean(np.array(data_sc)) - true))/len(true)
                        mae_pat_spec_ff_sc = sum(abs(pat_baseline_ff - true))/len(true)

                        dfs_cont[self.names_x0[index]].iloc[j]["count"] = len(recon) 
                        dfs_cont[self.names_x0[index]].iloc[j]["mae"] = mae.item()
                        dfs_cont[self.names_x0[index]].iloc[j]["mae_naive"] = mae_naive.item()
                        dfs_cont[self.names_x0[index]].iloc[j]["mae_pat_spec"] = mae_pat_spec_ff.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["count"] = len(recon) 
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["mae"] = mae_sc.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["mae_naive"] = mae_naive_sc.item()
                        dfs_cont_scaled[self.names_x0[index]].iloc[j]["mae_pat_spec"] = mae_pat_spec_ff_sc.item()
                    else:
                        dfs_cont[self.names_x0[index]].iloc[j]["count"] = 0
                        dfs_cont[self.names_x0[index]].iloc[j]["mae"] = np.nan
                        dfs_cont[self.names_x0[index]].iloc[j]["mae_naive"] = np.nan
                        dfs_cont[self.names_x0[index]].iloc[j]["mae_pat_spec"] = np.nan


                elif self.kinds_x0[index] != 'continuous':
                    all_targets = self.ground_truth_x[1][index][(self.non_missing_x_recon[:, elem:list_[index+1]] > 0).any(dim =1)]
                    mask_ = (self.non_missing_x_recon[to_keep,elem:list_[index+1]] > 0).any(dim =1)
                    recon =  self.res_list[index][0][to_keep]
                    true = self.ground_truth_x[1][index][to_keep][mask_]
                    pat_spec_ff = self.cat_baseline_x[1][index][to_keep][mask_]

                    recon = self.body.get_var_by_name(self.names_x0[index]).get_categories(recon)[mask_]
                    if len(recon) > 0:
                        acc = accuracy_score(true.flatten().astype(float), recon)
                        value, counts = np.unique(all_targets.flatten(), return_counts = True)
                        naive = np.random.choice(value, size = len(true.flatten()), p = counts/sum(counts))
                        naive_acc = accuracy_score(naive.astype(float), true.flatten().astype(float))
                        pat_spec_acc_ff = accuracy_score(pat_spec_ff, true.flatten().astype(float))
                        f1 = f1_score(true.flatten().astype(float), recon, average = "macro")
                        naive_f1 = f1_score(naive.astype(float), true.flatten().astype(float), average = "macro")
                        pat_spec_f1_ff = f1_score(pat_spec_ff, true.flatten().astype(float), average = "macro")


                        print(f'acc {acc}')
                        print(f'acc naive {naive_acc}')
                        #print(f'{classification_report(true.flatten().astype(float), recon)}')
                        dfs_cat[self.names_x0[index]].iloc[j]["acc"] = acc
                        dfs_cat[self.names_x0[index]].iloc[j]["naive acc"] = naive_acc
                        dfs_cat[self.names_x0[index]].iloc[j]["pat_spec"] = pat_spec_acc_ff
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["f1"] = f1
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["naive f1"] = naive_f1
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["pat_spec"] = pat_spec_f1_ff
                    else:
                        dfs_cat[self.names_x0[index]].iloc[j]["acc"] = np.nan
                        dfs_cat[self.names_x0[index]].iloc[j]["naive acc"] = np.nan
                        dfs_cat[self.names_x0[index]].iloc[j]["pat_spec"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["f1"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["naive f1"] = np.nan
                        dfs_cat_f1[self.names_x0[index]].iloc[j]["pat_spec"] = np.nan
                        
        self.df_res_cont = pd.DataFrame(np.nanmean(np.array([elem for elem in dfs_cont_scaled.values()], dtype = np.float64), axis =0), index = dfs_cont_scaled['Forced Vital Capacity (FVC - % predicted)'].index, columns = dfs_cont_scaled['Forced Vital Capacity (FVC - % predicted)'].columns)
        self.df_res_cat_acc = pd.DataFrame(np.nanmean(np.array([elem for elem in dfs_cat.values()], dtype = np.float64), axis =0), index = time_flags, columns = ["acc", "naive acc", "pat_spec"])
        self.df_res_cat_f1 = pd.DataFrame(np.nanmean(np.array([elem for elem in dfs_cat_f1.values()], dtype = np.float64), axis =0), index = time_flags, columns = ["f1", "naive f1", "pat_spec"])
        # plot
        fig, ax = plt.subplots()
        x_axis = range(len(self.df_res_cont))
        ax.plot(x_axis, self.df_res_cont.mae, label = 'Model')

        ax.plot(x_axis, self.df_res_cont.mae_pat_spec, '-.', label='previous value for patient')
        ax.plot(x_axis, self.df_res_cont.mae_naive, '-.', label = 'cohort mean')
        ax.set_title('Average accross all (scaled) MAE')
        ax.set_ylabel('MAE')
        ax.set_xlabel('years')
        ax.legend()
        
        fig, ax = plt.subplots()
        #x_axis = range(-len(tmp)+1, 1)
        x_axis = range(len(self.df_res_cat_acc))
        ax.plot(x_axis, self.df_res_cat_acc.acc, label = 'Model')

        ax.plot(x_axis, self.df_res_cat_acc.pat_spec, '-.', label='previous value for patient')
        ax.plot(x_axis, self.df_res_cat_acc["naive acc"], '-.', label = 'cohort mean')
        ax.set_title('Average accross all (scaled) accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('years')
        ax.legend()
        
        fig, ax = plt.subplots()
        #x_axis = range(-len(tmp)+1, 1)
        x_axis = range(len(self.df_res_cat_f1))
        ax.plot(x_axis, self.df_res_cat_f1.f1, label = 'Model')

        ax.plot(x_axis, self.df_res_cat_f1.pat_spec, '-.', label='previous value for patient')
        ax.plot(x_axis, self.df_res_cat_f1["naive f1"], '-.', label = 'cohort mean')
        ax.set_title('Average accross all (scaled) f1')
        ax.set_ylabel('f1')
        ax.set_xlabel('years')
        ax.legend()
        
    def get_result_df_y(self, time_flags = [(0,0), (0, 1), (1, 2), (2, 3), (3, 4),  (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12)]):
        list_ = np.cumsum([0] + self.splits_y0)
        df = {name: pd.DataFrame(index = time_flags, columns = ["acc", "f1", "acc_base", "f1_base"]) for name in self.names_y0}
        for j, interv in enumerate(time_flags):
            if interv[0] == interv[1]:
                to_keep = torch.tensor((self.delta_t_resc == interv[0]) & (self.num_rec_for_pred >= 0)).flatten()
            else:
                to_keep = torch.tensor((self.delta_t_resc > interv[0]) & (self.delta_t_resc <= interv[1])  & (self.num_rec_for_pred >= 0)).flatten()
            print(interv)
            for i, elem in enumerate(list_[:-1]):
                mask_ = (self.non_missing_y_recon[to_keep,list_[i]:list_[i+1]] > 0).any(dim =1)
                true_recon = self.ground_truth_y[0][to_keep, i][mask_]
                baseline_ = self.cat_baseline_y[to_keep, i][mask_]
                model_recon = self.predicted_cats_y[to_keep, i][mask_]
                name = self.names_y0[i]
                print(name)
                print(classification_report(true_recon, model_recon))
                print(classification_report(true_recon, baseline_))
                print(confusion_matrix(true_recon, model_recon))
                df[name].iloc[j]["acc"] = accuracy_score(true_recon, model_recon)
                df[name].iloc[j]["f1"] = f1_score(true_recon, model_recon, average = "macro")
                df[name].iloc[j]["acc_base"] = accuracy_score(true_recon, baseline_)
                df[name].iloc[j]["f1_base"] = f1_score(true_recon, baseline_, average = "macro")
        
        self.df_res_y = df
        # plot
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp.f1, label = 'ours')
            ax.plot(range(len(tmp)), tmp.f1_base, '-.', label='previous value for patient')
            ax.set_title(name)
            ax.set_ylabel('Macro F1')
            ax.set_xlabel('years')
            ax.legend()
        for index in range(len(self.names_y0)):
            name = self.names_y0[index]
            tmp = self.df_res_y[name]
            fig, ax = plt.subplots()
            ax.plot(range(len(tmp)), tmp.acc, label = 'ours')
            ax.plot(range(len(tmp)), tmp.acc_base, '-.', label='previous value for patient')
            ax.set_title(name)
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('years')
            ax.legend()
        
    def fill_tensor(self, data, mask, cat=False):
        # fill data 
        filled_data = torch.zeros_like(data)  # Initialize the filled tensor with zeros
        if cat:
            # intialize first row with some category
             filled_data[0] = self.create_one_tensor(data.shape[1])
        else:
            filled_data[0] = data[0]  # Copy the first row of data as it is

        # Iterate over columns and rows starting from the second row
        for i in range(1, data.size(0)):
            # fill with previous value if value is not available 
            filled_data[i] = torch.where(mask[i] == 1, data[i], filled_data[i - 1])

        return filled_data
    def create_one_tensor(self, n):
        tensor = torch.zeros(n)  # Initialize tensor with zeros
        tensor[0] = 1  # Set the value index to 1
        return tensor
    def create_random_tensor(self, n):
        tensor = torch.zeros(n)  # Initialize tensor with zeros
        random_index = random.randint(0, n - 1)  # Generate a random index within the range of n
        tensor[random_index] = 1  # Set the value at the random index to 1
        return tensor
    
