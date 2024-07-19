## FUNCTIONS - DATA- PLOT REG RSLT
from itertools import repeat, chain
from module_aladin.plot import *

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns
import os

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats("png2x")
# 테마 설정: "default", "classic", "dark_background", "fivethirtyeight", "seaborn"
mpl.style.use("fivethirtyeight")
# 이미지가 레이아웃 안으로 들어오도록 함
mpl.rcParams.update({"figure.constrained_layout.use": True})

import matplotlib.font_manager as fm
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
[fm.FontProperties(fname=font).get_name() for font in font_list if 'D2C' in font]
plt.rc('font', family='D2Coding')
mpl.rcParams['axes.unicode_minus'] = False

## FUNCTIONS - DATA- SCORE REG RSLT

def mase_nontime(actual,pred,base_pred) -> np.ndarray:
    """
    MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

    :param forecast: Forecast values. Shape: batch, time_o
    :param insample: Insample values. Shape: batch, time_i
    :param outsample: Target values. Shape: batch, time_o
    :param frequency: Frequency value
    :return: Same shape array with error calculated for each time step
    """
    return np.mean(np.abs(actual - pred)) / np.mean(np.abs(actual - base_pred),)

def make_reg_score_dict(y_actual,y_pred,base_val):
    rmse_model, rmse_base = np.sqrt(mse(y_actual,y_pred)), np.sqrt(mse([base_val]*len(y_actual),y_actual))
    mape_model, mape_base = mape(y_actual,y_pred), mape([base_val]*len(y_actual),y_actual)
    mase_model, mase_base = mase_nontime(y_actual,y_pred,base_val),1
    r2_model, r2_base = r2_score(y_actual,y_pred), 0
    
    return {
        'rmse' : [rmse_model, rmse_base],
        'mape' : [mape_model, mape_base],
        'mase' : [mase_model, mase_base],
        'r2_score' : [r2_model,r2_base]
    }

def print_reg_score_dict(name,dict_score,cut_line=True):
    print('{}\nr2 score : {:.5f}'.format(name,dict_score['r2_score'][0]))
    print('rmse_model : {:.5f} / rmse_base : {:.5f}\t'.format(*dict_score['rmse']),
          'mape_model : {:.5f} / mape_base : {:.5f}\t'.format(*dict_score['mape']))
    if cut_line : print('-'*150)

def make_reg_score_entire(dict_data,dict_rslt,print_rslt=False):
    dict_score = dict()
    for col, val in dict_data.items():
        X = val['X']
        y_actual = val['y']
        y_pred = dict_rslt[col]
        dict_score[col] = make_reg_score_dict(y_actual,y_pred,np.mean(y_actual))
        if print_rslt : print_reg_score_dict(col,dict_score[col])
    return dict_score

## PLOT

def scatter_reg_rslt(dict_data,dict_rslt,dict_score): #set_iput
    data_plot ={
        col : (dict_rslt[col]['valid'], data['valid']['y'])
        for col,data in dict_data.items() 
    }
    data_line ={
        col : (data['valid']['y'],data['valid']['y'])
        for col,data in dict_data.items() 
    }
#    fig,axes = plt.subplots(3,3,figsize=(12,12))
#    fig,axes = pair_plot_feat_hue(fig=fig,axes=axes,data=data_line,
    fig,axes = pair_plot_feat_hue(fig=None,axes=None,data=data_line,
                                  pair_plot=sns.lineplot,lw=0.3)
    #fig.set_size_inches(12,8, forward=True)
    fig,axes = pair_plot_feat_hue(fig=fig,axes=axes,data=data_plot,
                                  pair_plot=sns.scatterplot,s=5,alpha=0.65)

    for n,col in enumerate(data_plot.keys()):
        spprt_X = dict_data[col]['support']['X']
        ax = axes.flatten()[n] if len(dict_data) > 1 else  axes
        ax.set_ylabel('')
        ax.set_title(col,fontsize=12)
        ax.set_xlabel('rmse_model : {:.3f} | rmse_base : {:.3f}\n\
            r2 score : {:.3f} {:>35}\n'.format(*dict_score[col]['rmse'],
                                               dict_score[col]['r2_score'][0],
                                               f'n = {len(spprt_X)}'),
                      fontsize=10, ha ='left')
    ax_iter = axes.flatten() if len(dict_data) > 1 else [axes]
    for ax in ax_iter:
        plt.setp(ax.get_yticklabels(),rotation = 0, fontsize = 9)
        plt.setp(ax.get_xticklabels(),ha ='center',rotation = 0, fontsize = 9)
    
    return fig,axes

def plot_reg_score(dict_data,dict_rslt,dict_score):
    fig,axes = plt.subplots(len(dict_data),3,figsize=(15,4*len(dict_data)))
    for n, (col,data) in enumerate(dict_data.items()):
        ax_row = axes[n] if len(dict_data) > 1 else axes
        ax1, ax2, ax3 = ax_row[0], ax_row[1], ax_row[2]
        test_y, y_pred = (data['valid']['y'], dict_rslt[col]['valid'])
        train_y = data['train']['y']
        base_pred = np.mean(train_y)

        sns.histplot(test_y,label='actual',ax=ax1,alpha=0.5)
        sns.histplot(y_pred,label='knn_pred',ax=ax1,alpha=0.5)
        ax1.legend(fontsize=9)

        sns.histplot(test_y-base_pred,ax=ax2, label = 'baseline',alpha=0.5)
        sns.histplot(test_y-y_pred,ax=ax2, label = 'knn_pred',alpha=0.5)
        ax2.legend(fontsize=9)

        df_score = pd.DataFrame(dict_score[col]).T[[1,0]]
        xs = list(chain.from_iterable(repeat(val,2) for val in df_score.index))
        ax3r = ax3.twinx()
        sns.barplot(x=xs[:2],y=list(df_score.values.reshape(-1))[:2],
                    hue = ['knn_pred','base']*1,ax=ax3,alpha=0.65,legend=False)
        sns.barplot(x=xs[2:],y=list(df_score.values.reshape(-1))[2:],
                    hue = ['knn_pred','base']*3,ax=ax3r,alpha=0.8,legend=False)
        ax3.set_yscale('log')
        ax3r.set_ylim([0.0,1.15])
        ax3r.bar_label(ax3r.containers[0], fontsize=8, fmt='%.4f')
        ax3r.bar_label(ax3r.containers[1], fontsize=8, fmt='%.4f')
        ax3.grid(False)
        ax3r.grid(False)
        ax3r.set_yscale('linear')    
        ax3.bar_label(ax3.containers[0], fontsize=8, fmt='%.4f')
        ax3.bar_label(ax3.containers[1], fontsize=8, fmt='%.4f')

        ax1.set_title(str_cutter(col,50),fontsize=15,loc='left',ha='left')
        ax2.xaxis.set_label_coords(-0.02, -0.15)
        ax2.set_xlabel('rmse_model : {:.3f} | rmse_base : {:.3f}\nr2 score : {:.3f} {:>48}\n'.format(*dict_score[col]['rmse'],dict_score[col]['r2_score'][0],
            f'n = {len(train_y)}'), fontsize=10,ha ='left')
        ax1.set_xlabel('')
        ax3.set_xlabel('')
        ax1.set_ylabel('count',fontsize =10) 
        ax2.set_ylabel('count',fontsize =10) 
        plt.setp(ax3.get_yticklabels(),rotation = 0, fontsize = 9, color='#333333')
        plt.setp(ax3r.get_yticklabels(),rotation = 0, fontsize = 9)

    for ax in axes.flatten():
        plt.setp(ax.get_yticklabels(),rotation = 0, fontsize = 9)
        plt.setp(ax.get_xticklabels(),ha ='center',rotation = 0, fontsize = 9)

    return fig, axes

def plot_n_save_regrslt(save_dir,work_name,dict_data,dict_rslt,dict_score,notice=True):
    if not os.path.exists(save_dir): os.mkdir(save_dir)

    fig,axes = scatter_reg_rslt(dict_data,dict_rslt,dict_score)
    file_name = 'reg_scatter_{}.png'.format(work_name)
    fig.savefig(os.path.join(save_dir,file_name))
    fig,axes = plot_reg_score(dict_data,dict_rslt,dict_score)
    file_name = 'reg_rslt_{}.png'.format(work_name)
    fig.savefig(os.path.join(save_dir,file_name))
    if notice : print("plot completed")