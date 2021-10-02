import pandas as pd
import matplotlib as plt
import streamlit as st
class plot_wells_status_purpose_content:
    def plot(self,df_wells,userValue,final_directory):
        fluidsListPR = df_wells[df_wells['wlbStatus'] == 'PRODUCING']['wlbContent'].value_counts().index.to_list()
        fluidsListING = df_wells[df_wells['wlbStatus'] == 'INJECTING']['wlbContent'].value_counts().index.to_list()
        # Show Status Hist
        # with st.beta_expander("Click here to show well's status histograms",False):
        col1, col2, col3 = st.beta_columns(3)

        # get oil and gas series
        def getSeries(status, fluid):
            oilPRODUCINGdict = df_wells[df_wells['wlbContent'] == fluid]['wlbStatus'][
                df_wells[df_wells['wlbContent'] == fluid]['wlbStatus'] == status].value_counts()
            oilPRODUCINGdict = {status + ' ' + fluid: oilPRODUCINGdict[0]}
            oilPRODUCINGdict = pd.Series(oilPRODUCINGdict)
            return oilPRODUCINGdict

        if df_wells['wlbContent'].value_counts().shape[0] > 0:
            ax = df_wells['wlbContent'].value_counts().plot(kind='bar', figsize=(6, 4.5));
            ax.bar_label(ax.containers[0]);

            plt.xticks(fontsize=5.5, rotation=0)
            plt.ylabel('Number of Wells')
            plt.title(userValue + ' Wellbore Content');
            plt.savefig(final_directory + '/' + userValue + " Wellbore Content Histogram.png")
            col2.pyplot()

            # get oil, gas, water series (call the fun) and append the oil,gas,water
            if 'OIL' in fluidsListPR:
                oilPRODUCINGdser = getSeries('PRODUCING', 'OIL')
                statusHist = df_wells['wlbStatus'].value_counts().append(oilPRODUCINGdser)
            if 'GAS' in fluidsListPR:
                gasPRODUCINGdser = getSeries('PRODUCING', 'GAS')
                try:
                    statusHist = statusHist.append(gasPRODUCINGdser)
                except:
                    statusHist = df_wells['wlbStatus'].value_counts().append(gasPRODUCINGdser)
            if 'WATER' in fluidsListING:
                waterINJECTINGGdser = getSeries('INJECTING', 'WATER')
                try:
                    statusHist = statusHist.append(waterINJECTINGGdser)
                except:
                    statusHist = df_wells['wlbStatus'].value_counts().append(waterINJECTINGGdser)
            if 'GAS' in fluidsListING:
                gasINJECTINGdser = getSeries('INJECTING', 'GAS')
                try:
                    statusHist = statusHist.append(gasINJECTINGdser)
                except:
                    statusHist = df_wells['wlbStatus'].value_counts().append(gasINJECTINGdser)

            statusHist = statusHist.sort_values(ascending=False)

            statusHist = statusHist.reset_index()
            statusHist = statusHist.sort_values('index')
            statusHist = statusHist.set_index('index')
            fluidsStatList = statusHist.index.to_list()

            # plot the statusHist
            p = statusHist.plot(kind='bar', figsize=(8, 4.5), legend=False);
            p.bar_label(p.containers[0]);

            if 'PRODUCING OIL' in fluidsStatList:
                oilIndex = fluidsStatList.index('PRODUCING OIL')
                p.patches[oilIndex].set_color('green')
            if 'PRODUCING GAS' in fluidsStatList:
                GASIndex = fluidsStatList.index('PRODUCING GAS')
                p.patches[GASIndex].set_color('red')
            if 'INJECTING WATER' in fluidsStatList:
                WATERIndex = fluidsStatList.index('INJECTING WATER')
                p.patches[WATERIndex].set_color('#d4f1f9')
            if 'INJECTING GAS' in fluidsStatList:
                GASIndex = fluidsStatList.index('INJECTING GAS')
                p.patches[GASIndex].set_color('red')

            plt.xticks(fontsize=4, rotation=0)
            plt.ylabel('Number of Wells')
            plt.xlabel('')
            plt.title(userValue + ' Wellbore Status');
            plt.savefig(final_directory + '/' + userValue + " Wellbore Status Histogram2.png")
            st.pyplot()