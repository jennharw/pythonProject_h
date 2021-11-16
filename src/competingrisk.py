import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
import torchtuples as tt # Some useful functions
from pycox.models.deephit import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

import time

class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')

# NetworkArchitecture
class CauseSpecificNet(torch.nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                 out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
            )
            self.risk_nets.append(net)

    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out
    
np.random.seed(1234)
_ = torch.manual_seed(1234)

class DeepHitCompetingRisk:
    def __init__(self, graduate):
        self.graduate = graduate

        #eval
        self.x_test2 = 0
        self.durations_test =0
        self.events_test = 0

    def training(self):
        #휴학 오류
        graduate1 = self.graduate[self.graduate['휴학횟수'] > 6]
        graduate1['휴학횟수'] = 1
        graduate1['휴학기간'] = 1
        graduate = self.graduate[self.graduate['휴학횟수'] < 6]
        self.graduate = pd.concat([graduate1, graduate], axis=0)
        plt.boxplot(self.graduate['휴학횟수'])
        plt.title("휴학Dropout box plot")
        plt.show()


        cpSet = self.graduate

        cpSet = cpSet[['기간', 'event', '자타', 'count', '인건비합', '등록금장학', 'etc_장학', '성적','입학성적', '휴학기간']]
        cpSet['기간'] = cpSet['기간'].astype('int64')
        cpSet['자타'] = cpSet['자타'].astype('float64')
        print(cpSet['event'].max())

        df_train = cpSet
        df_test = df_train.sample(frac=0.1)
        df_train = df_train.drop(df_test.index)
        df_val = df_train.sample(frac=0.1)
        df_train = df_train.drop(df_val.index)
        get_x = lambda df: (df
                            .drop(columns=['기간', 'event'])
                            .values.astype('float32'))
        x_train = get_x(df_train)
        x_val = get_x(df_val)
        x_test = get_x(df_test)
        self.x_test2 = x_test

        #Label Transform
        num_durations = 6
        labtrans = LabTransform(num_durations)
    
        get_target = lambda df: (df['기간'].values.astype('int64'), df['event'].values)
        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))
        durations_test, events_test = get_target(df_test)
        self.durations_test = durations_test
        self.events_test = events_test

        return
        val = (x_val, y_val)
    
        
        in_features = x_train.shape[1]
        num_nodes_shared = [64, 64]
        num_nodes_indiv = [32]
        num_risks = y_train[1].max()
        out_features = len(labtrans.cuts)
        batch_norm = True
        dropout = 0.1
    
        # net = SimpleMLP(in_features, num_nodes_shared, num_risks, out_features)
        net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                               out_features, batch_norm, dropout)
    
        optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01,
                                    cycle_eta_multiplier=0.8)
        model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1,
                        duration_index=labtrans.cuts)
    
        epochs = 512
        batch_size = 256
        callbacks = [tt.callbacks.EarlyStoppingCycle()]
        verbose = False  # set to True if you want printout

        print("-----Training------")
        start = time.time()
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=val)
        _ = log.plot()
        plt.show()
        print(f'Time:{time.time()-start}')
        model.save_model_weights('model/dhcr.pkl')
        print(">>>>>>>>>>>>>>>>>>>>>>>>>")

        return model, labtrans.cuts

    def predict_competingrisk(self, dept_cd="경영학과"):

        in_features = 8
        num_nodes_shared = [64, 64]
        num_nodes_indiv = [32]
        num_risks = 2
        out_features = 6
        batch_norm = True
        dropout = 0.1
        net = CauseSpecificNet(in_features, num_nodes_shared, num_nodes_indiv, num_risks,
                               out_features, batch_norm, dropout)

        model = DeepHit(net)
        model.load_model_weights('model/dhcr.pkl')

        test_set = self.graduate[
            (self.graduate['rec014_ent_year'] == '2021') & (self.graduate['rec014_ent_term'] == '1R') & (
                    self.graduate['학과'] == dept_cd)]
        test_set['rec014_std_id'] = test_set['rec014_std_id'].astype('string')
        print("-------------test---------------")
        x_test = test_set[['자타', 'count', '인건비합', '등록금장학','etc_장학', '성적','입학성적', '휴학기간', '기간', 'event', 'rec014_std_id']]
        get_x = lambda df: (df
                            .drop(columns=['기간', 'event', 'rec014_std_id'])
                            .values.astype('float32'))
        x_test = get_x(x_test)

        # predict

        #legend 제적cif 1 가 높은 것

        cif = model.predict_cif(x_test)
        test_set['cif_drop'] = cif[1][5] #Transpose 해서 넣기

        test_set = test_set.reset_index()
        hazard_students = test_set.sort_values(by = 'cif_drop', ascending=True).head(int(len(test_set)*0.1)) # 상위 ?

        so = math.ceil(len(hazard_students) / 3)
        fig, axs = plt.subplots(so, 3, True, True, figsize=(5 * so, 5))
        for ax, idx in zip(axs.flat, hazard_students.index.tolist()):
            #if test_set.iloc[idx]['rec014_std_id'] in hazard_students['rec014_std_id'].tolist():
            mylabel = "std %s"%(str(test_set.iloc[idx]['rec014_std_id']))
            #else:
            #    mylabel = None
            ax.plot(pd.DataFrame(cif.transpose()[idx], index=[0. , 1.2 ,2.4, 3.6, 4.8 ,6. ]), label= [None, mylabel]) #labtrans cuts
            ax.set_ylabel('CIF')
            ax.set_xlabel('Time')
            ax.legend()
            ax.grid(linestyle='--')
        plt.show()
        #pd.set_option('display.max_columns', 40)
        #pd.set_option('display.width', 1000)
        #print(test_set)
        #print(hazard_students)

        # EVALUATION
        from pycox.evaluation import EvalSurv
        print("C-index")
        print(self.durations_test)
        print(self.events_test)
        print(self.x_test2)
        surv = model.predict_surv_df(self.x_test2)
        ev = EvalSurv(surv, self.durations_test, self.events_test != 0, censor_surv='km')
        print("DeepHit with competing risks - C-index", ev.concordance_td())

        #cumulative incidence function
        cif = model.predict_cif(self.x_test2)
        cif1 = pd.DataFrame(cif[0], model.duration_index)
        cif2 = pd.DataFrame(cif[1], model.duration_index)
        ev1 = EvalSurv(1 - cif1, self.durations_test, self.events_test == 1, censor_surv='km')
        ev2 = EvalSurv(1 - cif2, self.durations_test, self.events_test == 2, censor_surv='km')
        print("DeepHit cumulative incidence function c-index", ev1.concordance_td())
        print("DeepHit CIF event 2, c-index", ev2.concordance_td())


        return fig, hazard_students



