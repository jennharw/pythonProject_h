import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

import torch # For building the networks
import torchtuples as tt # Some useful functions
from pycox.models.deephit import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

import os


class survivalSimple:
    def __init__(self, graduate):

        self.graduate = graduate
        self.in_features = 0
        self.out_features = 0

        
    def simplemodel(self):
        DHSset = self.graduate
        # graduate['자타'] = graduate.apply(lambda x: 1 if (x['학부출신'] == '고려대학교') else 0, axis=1)
        DHSset = DHSset[['기간', 'event', 'count', '인건비합','등록금장학' ,'etc_장학', '성적','입학성적' ,'휴학기간']]
        DHSset = DHSset.fillna(0)
        DHSset = DHSset.iloc[:2000]
        DHSset['기간'] = DHSset['기간'].astype('int64')
        #DHSset['자타'] = DHSset['자타'].astype('float64')
        print(DHSset['event'].max())

        np.random.seed(1234)
        _ = torch.manual_seed(1234)

        df_train = DHSset
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
    
        class LabTransform(LabTransDiscreteTime):
            def transform(self, durations, events):
                durations, is_event = super().transform(durations, events > 0)
                events[is_event == 0] = 0
                return durations, events.astype('int64')
    
        num_durations = 6
        labtrans = LabTransform(num_durations)
    
        get_target = lambda df: (df['기간'].values.astype('int64'), df['event'].values)

        y_train = labtrans.fit_transform(*get_target(df_train))
        y_val = labtrans.transform(*get_target(df_val))
        durations_test, events_test = get_target(df_test)
        val = (x_val, y_val)
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
    
        #nueral net
        self.in_features = x_train.shape[1]
        num_nodes = [32, 32]
        self.out_features = 6
        batch_norm = True
        dropout = 0.1
    
        #net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
        net = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),
    
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),
    
            torch.nn.Linear(32, self.out_features)
        )
    
        #training
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
        np.random.seed(1234)
        _ = torch.manual_seed(123)
    
        print("------training--------")
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name(0))
        torch.cuda.device(0)
       
    
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
        batch_size = 256
        lr_finder = model.lr_finder(x_train, y_train, batch_size, tolerance=3)
        _ = lr_finder.plot()
        plt.show()
    
        print(lr_finder.get_best_lr())
    
        model.optimizer.set_lr(0.01)
    
        epochs = 100
        callbacks = [tt.callbacks.EarlyStopping()]
        log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
        _ = log.plot()
        plt.show()
    
        
        
        ## eval
        # ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        # ev.concordance_td('antolini')
        # time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
        # ev.brier_score(time_grid).plot()
        # plt.ylabel('Brier score')
        # _ = plt.xlabel('Time')
        
        # 저장
        #print(model.state_dict())
        model.save_model_weights('model/dhs.pkl')
        return model
        
    def predict_by_dept(self, dept_cd = '경영학과'):
        #predict
        torch.cuda.device(0)

        net = torch.nn.Sequential(
            torch.nn.Linear(7, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.1),

            torch.nn.Linear(32, 6)
        )
        #net.load_state_dict(torch.load('model/dhs.pkl'))
        #print(torch.load('model/dhs.pkl'))

        model = DeepHitSingle(net)#.load_model_weights('model/dhs.pkl')
        model.load_model_weights('model/dhs.pkl')

        test_set = self.graduate[
            (self.graduate['rec014_ent_year'] == '2021') & (self.graduate['rec014_ent_term'] == '1R') & (
                        self.graduate['학과'] == dept_cd)]
        test_set['rec014_std_id'] = test_set['rec014_std_id'].astype('string')

        x_test = test_set[['자타', 'count', '인건비합','등록금장학' ,'etc_장학', '성적','입학성적', '휴학기간','기간', 'event','rec014_std_id', '과정']]
        get_x = lambda df: (df
                            .drop(columns=['자타','기간', 'event','rec014_std_id','과정'])
                            .values.astype('float32'))
        x_test = get_x(x_test)

        #predict
        surv = model.predict_surv_df(x_test)
        test_set = pd.concat([test_set.reset_index(), surv.iloc[1]], axis = 1)

        hazard_students = test_set.sort_values(by = 1, ascending=True).head(int(len(test_set)*0.1)) # 상위 ?
        fig3 = plt.figure()
        ax = fig3.add_subplot(111)
        for i in range(len(test_set)):
            if test_set.iloc[i]['rec014_std_id'] in hazard_students['rec014_std_id'].tolist():
                mylabel = "std %s"%(str(test_set.iloc[i]['rec014_std_id']))
            else:
                mylabel = None
            ax.plot(surv.iloc[:, i], label = mylabel)

        #plt.ylabel('S(t | x)')
        #_ = plt.xlabel('Time')
        #plt.legend()
        #plt.show()
        #Sort
        #print(test_set.sort_values(by = 1.2, ascending=True))
        return fig3, hazard_students

        # surv = model.interpolate(10).predict_surv_df(x_test)
        # print(x_test[:5])
        # surv.iloc[:, :5].plot(drawstyle='steps-post')
        # plt.ylabel('S(t | x)')
        # _ = plt.xlabel('Time')
        # plt.show()

    


