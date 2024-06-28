from statistical_analysis import  StatisticalAnalysis
import numpy as np
import os

from systematics import systematics as sys
import matplotlib.pyplot as plt


current_file = os.path.dirname(os.path.abspath(__file__))

class Model:

    def __init__(self, get_train_set=None, systematics=None):

        self.train_set = get_train_set
        self.systematics = systematics

        del self.train_set["settings"]

        print("Full data: ", self.train_set["data"].shape)
        print("Full Labels: ", self.train_set["labels"].shape)
        print("Full Weights: ", self.train_set["weights"].shape)
        print("sum_signal_weights: ", self.train_set["weights"][self.train_set["labels"] == 1].sum(),)
        print("sum_bkg_weights: ", self.train_set["weights"][self.train_set["labels"] == 0].sum(),)
        print(" \n ")

        self.training_set, self.valid_set = train_test_split(data_set=self.train_set, test_size=0.5, random_state=42, reweight=True)

        del self.train_set

        print("Training Data: ", self.training_set["data"].shape)
        print("Training Labels: ", self.training_set["labels"].shape)
        print("Training Weights: ", self.training_set["weights"].shape)
        print("sum_signal_weights: ", self.training_set["weights"][self.training_set["labels"] == 1].sum(),)
        print("sum_bkg_weights: ", self.training_set["weights"][self.training_set["labels"] == 0].sum(),)
        print()
        print("Valid Data: ", self.valid_set["data"].shape)
        print("Valid Labels: ", self.valid_set["labels"].shape)
        print("Valid Weights: ", self.valid_set["weights"].shape)
        print("sum_signal_weights: ", self.valid_set["weights"][self.valid_set["labels"] == 1].sum(),)
        print("sum_bkg_weights: ", self.valid_set["weights"][self.valid_set["labels"] == 0].sum(),)
        print(" \n ")
        print("Training Data: ", self.training_set["data"].shape)

        self.re_train = True

        from boosted_decision_tree import BoostedDecisionTree
        self.model = BoostedDecisionTree()
        module_file = current_file + "/model_XGB.json"
        if os.path.exists(module_file):
            self.model.load(module_file)
            self.re_train = False  # if model is already trained, no need to retrain
        self.name = "model_XGB"
        print("Model is BDT")
        self.stat_analysis = StatisticalAnalysis(self.model, self.valid_set)

    def fit(self):

        if self.re_train:
            balanced_set = self.balance_set()
            self.model.fit(balanced_set["data"], balanced_set["labels"], balanced_set["weights"])
            self.model.save( current_file + "/" + self.name)

        saved_info_file = current_file + "/saved_info_" + self.name + ".pkl"
        if os.path.exists(saved_info_file):
            self.stat_analysis.load(saved_info_file) 
        else:   
            self.stat_analysis.calculate_saved_info()
            self.stat_analysis.save(saved_info_file)

        train_score = self.model.predict(self.training_set["data"])
        train_results = self.stat_analysis.compute_mu(train_score, self.training_set["weights"],plot=True)
       
        print("Train Results: ")
        for key in train_results.keys():
            print("\t", key, " : ", train_results[key])
        
        mode = 1 #To switch between systematics study
        NLL = True #To switch between NLL and bins method for mu calculation
        plt.figure()
        
        if mode == 1: #We change one systematic at time

            mu_syst={}
            Bornes_syst={"tes":[0.9,1.1,2],"jes":[0.9,1.1,2],"soft_met":[0,5,2],"bkd_scale":[0.5,2,2],"ttbar_scale":[0.5,2,2],"diboson_scale":[0.5,2,2]}
            for syst,borne in Bornes_syst.items():
                liste_mu_syst=[]
                for x in np.linspace(borne[0],borne[1],borne[2]):
                    if syst=="tes":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=x, jes=1.0, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=None, verbose=0,)
                    elif syst=="jes":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=1.0, jes=x, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=None, verbose=0,)
                    elif syst=="soft_met":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=1.0, jes=1.0, soft_met=x, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=None, verbose=0,)
                    elif syst=="bkd_scale":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=1.0, jes=1.0, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=x, verbose=0,)
                    elif syst=="ttbar_scale":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=1.0, jes=1.0, soft_met=0, seed=31415, ttbar_scale=x, diboson_scale=None, bkg_scale=None, verbose=0,)
                    elif syst=="diboson_scale":
                        train_plus_syst = sys(data_set=self.valid_set.copy(), tes=1.0, jes=1.0, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=x, bkg_scale=None, verbose=0,)
    
                    valid_score_syst= self.model.predict(train_plus_syst["data"])
                    valid_results_syst= self.stat_analysis.compute_mu(valid_score_syst, train_plus_syst["weights"], syst_value=x, which_syst=syst, NLL=NLL)
                    print("Valid_syst Results: ")
                    for key in valid_results_syst.keys():
                        if key=="mu_hat":
                            liste_mu_syst.append([x,valid_results_syst[key]])
                            mu_syst[syst]=liste_mu_syst
                        
                        print("\t", key, " : ", valid_results_syst[key])
                    print(f"{syst}={x}")
                
            fig, axes = plt.subplots(2, 3, figsize=(20, 5), sharey=True)
    
            for idx, (key, data) in enumerate(mu_syst.items()):
                x_values = [point[0] for point in data]
                y_values = [point[1] for point in data]
                abs_y_values = [abs(y) for y in y_values]
                
                row, col = idx//3, idx%3
                axes[row, col].plot(x_values, abs_y_values, marker='o', linestyle='-', color='b')
                axes[row, col].set_title(key)
                if idx >=3:
                    axes[row, col].set_xlabel('Syst value')
                if idx == 0 or idx==3:
                    axes[row, col].set_ylabel('mu value')
                axes[row, col].grid(True)
    
            plt.tight_layout()
            plt.savefig('systematics.png')
            plt.show()
        
        elif mode == 2: #We change both tes and bkd_scale
            
            Bornes_syst = {"tes": [0.9, 1.1, 5], "bkd_scale": [0.5, 2, 4]}

            tes_values = np.linspace(Bornes_syst["tes"][0], Bornes_syst["tes"][1], Bornes_syst["tes"][2])
            bkd_scale_values = np.linspace(Bornes_syst["bkd_scale"][0], Bornes_syst["bkd_scale"][1], Bornes_syst["bkd_scale"][2])

            global_mu = []
            for i, x in enumerate(bkd_scale_values):
                local_mu = []
                for j, y in enumerate(tes_values):
                    train_plus_syst = sys(data_set=self.valid_set.copy(), tes=y, jes=1.0, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=x, verbose=0)

                    valid_score_syst = self.model.predict(train_plus_syst["data"])
                    valid_results_syst = self.stat_analysis.compute_mu(valid_score_syst, train_plus_syst["weights"], NLL=NLL)
            
                    print("Valid_syst Results: ")
                    for key in valid_results_syst.keys():
                        if key == "mu_hat":
                            local_mu.append(valid_results_syst[key])
                
                        print("\t", key, " : ", valid_results_syst[key])
                    print(f"tes={y}, bkd_scale={x}")
                global_mu.append(local_mu)

            for i in range(len(bkd_scale_values)):
                plt.plot(tes_values,global_mu[i], marker='o', linestyle='-', label='bkd_scale = '+str(bkd_scale_values[i]))
                plt.xlabel("tes")
                plt.ylabel("$\hat{\mu}$")
                plt.legend()

            plt.savefig('systematics.png')
            plt.show()
        
        elif mode == 3: #We change tes and bkd_scale but let them fix and repeat several times mu calculation to observe issues with the code
            
            tes_value, bkd_scale_value = 1.1, 2
            for _ in range(3):
                train_plus_syst = sys(data_set=self.valid_set.copy(), tes=tes_value, jes=1.0, soft_met=0, seed=31415, ttbar_scale=None, diboson_scale=None, bkg_scale=bkd_scale_value, verbose=0)
        
                valid_score_syst = self.model.predict(train_plus_syst["data"])
                valid_results_syst = self.stat_analysis.compute_mu(valid_score_syst, train_plus_syst["weights"], NLL=NLL)
                
                print("Valid_syst Results: ")
                for key in valid_results_syst.keys():
                    print("\t", key, " : ", valid_results_syst[key])
                print(f"tes={tes_value}, bkd_scale={bkd_scale_value}")

    def balance_set(self):
        balanced_set = self.training_set.copy()

        weights_train = self.training_set["weights"].copy()
        train_labels = self.training_set["labels"].copy()
        class_weights_train = (weights_train[train_labels == 0].sum(), weights_train[train_labels == 1].sum(),)

        for i in range(len(class_weights_train)):  # loop on B then S target
            # training dataset: equalize number of background and signal
            weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
            )
            # test dataset : increase test weight to compensate for sampling

        balanced_set["weights"] = weights_train

        return balanced_set

    def predict(self, test_set):

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        predictions = self.model.predict(test_data)

        result = self.stat_analysis.compute_mu(predictions, test_weights)

        print("Test Results: ", result)

        return result

def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)
    
    print(f"Full size of the data is {full_size}")
    
    np.random.seed(random_state)
    if isinstance(test_size, float):
        test_number = int(test_size * full_size)
        random_index = np.random.randint(0, full_size, test_number)
    elif isinstance(test_size, int):
        random_index = np.random.randint(0, full_size, test_size)
    else:
        raise ValueError("test_size should be either float or int")

    full_range = data.index
    remaining_index = full_range[np.isin(full_range, random_index, invert=True)]
    remaining_index = np.array(remaining_index)
    
    print(f"Train size is {len(remaining_index)}")
    print(f"Test size is {len(random_index)}")
    
    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            array = np.array(data_set[key])
            test_set[key] = array[random_index]
            train_set[key] = array[remaining_index]

    test_set["data"] = data.iloc[random_index]
    train_set["data"] = data.iloc[remaining_index]

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
            train_set["labels"] == 1
        ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
            test_set["labels"] == 1
        ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
            train_set["labels"] == 0
        ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
            test_set["labels"] == 0
        ] * (background_weight / background_weight_test)

    return train_set, test_set