import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import streamlit as st
import os
import zipfile

from app import zipdir, plotMult1, plot_multi3, plot_multi2, get_binary_file_downloader_html


class group_plot:
    def plot(self,userValues,userValue,uniteType_Oil,graphNum,dfMultOil,final_directory,df_new,dft_new,answer,csumNames,mcolors,df_newcSUM,dftt_newcSUM,yearsx):
        # =====================================MultiOil=======================================================
        if ('OIL' in userValues):
            if uniteType_Oil == 'STB':
                dfMultOil = dfMultOil * 6.2898

            years = mdates.YearLocator()  # every year
            months = mdates.MonthLocator()  # every month
            years_fmt = mdates.DateFormatter('%Y')

            yearsxoil = dfMultOil.index.year.to_list()
            yearsxoil = list(set(yearsxoil))

            ax = dfMultOil.plot(figsize=(20, 10), x_compat=True);

            for year in yearsxoil:
                plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)
            plt.title('Oil Production');
            plt.xlabel('Years');
            if uniteType_Oil == 'STB':
                plt.ylabel('Production Rate (STB/Month)');
            else:
                plt.ylabel('Production Rate (MSm3/Month)');

            # format the ticks
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(years_fmt)
            ax.xaxis.set_minor_locator(months)

            # round to nearest years.
            datemin = np.datetime64(dfMultOil.index[0], 'Y')
            datemax = np.datetime64(list(dfMultOil.index)[-2], 'Y') + np.timedelta64(1, 'Y')
            ax.set_xlim(datemin, datemax)

            ax.grid(axis='both', which='both')
            plt.savefig(final_directory + '/' + ' multiple fields oil rate year.png')
            st.pyplot()

            # months indexes
            dfMultOilShifted = dfMultOil.copy()
            dfMultOilShifted = dfMultOilShifted.apply(lambda x: pd.Series(x.dropna().values))

            # plot
            ax = dfMultOilShifted.reset_index(drop=True).plot(figsize=(20, 10), x_compat=True);
            plt.xlabel('Months');

            for tick in np.arange(0, dfMultOilShifted.shape[0] + 1, 12):
                plt.axvline(tick, color='black', linewidth=1)

            minor_locator = FixedLocator(dfMultOilShifted.reset_index(drop=True).index.to_list())
            ax.xaxis.set_minor_locator(minor_locator)

            plt.title('Oil Production');
            # plt.xlabel('Time');
            if uniteType_Oil == 'STB':
                plt.ylabel('Production Rate (STB/Month)');
            else:
                plt.ylabel('Production Rate (MSm3/Month)');

            # round
            datemin = 0
            datemax = dfMultOilShifted.shape[0]
            ax.set_xlim(datemin, datemax)

            plt.xticks(np.arange(0, dfMultOilShifted.shape[0] + 1, 12))

            ax.grid(axis='both', which='both')
            plt.savefig(final_directory + '/' + ' multiple fields oil rate month.png')
            st.pyplot()

        # ============================================================================================

        if len(graphNum) != 1:
            # ploting with Fluid Production
            plotMult1(df_new, dft_new, 'yes')

            # Trim Oil date Graph
            if ('OIL' in userValues):
                plotMult1(df_new, dft_new, 'no')

        def plotMult2(df_newcSUM, dftt_newcSUM, xtime):
            # ploting time with Fluid Production
            if len(graphNum) != 1:
                if (answer == 'group' or answer == 'both'):
                    if xtime == 'yes':
                        years = mdates.YearLocator()  # every year
                        months = mdates.MonthLocator()  # every month
                        years_fmt = mdates.DateFormatter('%Y')

                        colors2 = ['green', 'red', 'orange', 'black', 'blue']
                        ax = df_newcSUM[csumNames].set_index('Years').plot(figsize=(25, 10), color=mcolors,
                                                                           x_compat=True)
                        plt.ylabel('Fluid Cumulative Production (MSm3)');

                        for year in yearsx:
                            plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)

                        if 'GAS' in userValues:
                            ax2 = ax.twinx()
                            # make a plot with different y-axis using second axis object
                            ax2.plot(dftt_newcSUM['GAS Cumulative Production'], label="GAS", color="red");
                            ax2.set_ylabel("GAS Cumulative Production (BSm3)")
                            plt.legend(loc=(0.95, 1))

                        # format the ticks
                        ax.xaxis.set_major_locator(years)
                        ax.xaxis.set_major_formatter(years_fmt)
                        ax.xaxis.set_minor_locator(months)

                        # round to nearest years.
                        datemin = np.datetime64(df_newcSUM['Years'][0], 'Y')
                        datemax = np.datetime64(list(df_newcSUM['Years'])[-2], 'Y') + np.timedelta64(1, 'Y')
                        ax.set_xlim(datemin, datemax)

                        ax.grid(axis='both', which='both')

                        plt.title(str(userValue) + ' Field Cumulative Production');
                        plt.savefig(final_directory + '/' + userValue + ' field cumulative production year.png')
                        st.pyplot()

                    else:
                        from matplotlib.ticker import FixedLocator

                        colors2 = ['green', 'orange', 'black', 'blue']
                        ax = df_newcSUM[csumNames].plot(figsize=(25, 10), color=mcolors, x_compat=True)
                        plt.ylabel('Fluid Cumulative Production (MSm3)');
                        plt.xlabel('Months');

                        for tick in np.arange(0, df_newcSUM.shape[0] + 1, 12):
                            plt.axvline(tick, color='black', linewidth=1)

                        if 'GAS' in userValues:
                            ax2 = ax.twinx()
                            # make a plot with different y-axis using second axis object
                            ax2.plot(dftt_newcSUM.reset_index()['GAS Cumulative Production'], label="GAS", color="red");
                            ax2.set_ylabel("GAS Cumulative Production (BSm3)")
                            plt.legend(loc=(0.95, 1))

                        # format the ticks
                        # minor_locator = AutoMinorLocator(2)
                        minor_locator = FixedLocator(df_newcSUM.index.to_list())
                        ax.xaxis.set_minor_locator(minor_locator)

                        # round
                        datemin = 0
                        datemax = df_newcSUM.shape[0]
                        ax.set_xlim(datemin, datemax)

                        plt.xticks(np.arange(0, df_newcSUM.shape[0] + 1, 12))

                        ax.grid(axis='both', which='both')

                        plt.title(str(userValue) + ' Field Cumulative Production');
                        plt.savefig(final_directory + '/' + userValue + ' field cumulative production month.png')
                        st.pyplot()

        #  ploting time with Fluid Production
        plotMult2(df_newcSUM, dftt_newcSUM, 'yes')

        # Trim Oil date Graph
        if ('OIL' in userValues):
            plotMult2(df_newcSUM, dftt_newcSUM, 'no')

        def plotMultiy1(dft_new, xtime):
            userValuesclr = userValues.copy()
            if len(graphNum) != 1:
                if (answer == 'group' or answer == 'both'):
                    if xtime == 'yes':
                        for year in yearsx:
                            plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)
                        plot_multi3(dft_new, userValuesclr, xtime, figsize=(25, 10));

                        plt.title(str(userValue) + ' Field Production');
                        plt.savefig(final_directory + '/' + userValue + ' field production year multy.png')
                        st.pyplot()
                    else:
                        for tick in np.arange(0, dft_new.shape[0] + 1, 12):
                            plt.axvline(tick, color='black', linewidth=1)

                        plot_multi3(dft_new, userValuesclr, xtime, figsize=(25, 10));

                        plt.title(str(userValue) + ' Field Production');
                        plt.savefig(final_directory + '/' + userValue + ' field production month multy.png')
                        st.pyplot()

        # ploting time with Fluid Production (Multiple y-axis)
        plotMultiy1(dft_new, 'yes')

        # ploting time with Fluid Production (Multiple y-axis)(indexes)
        if ('OIL' in userValues):
            plotMultiy1(dft_new.reset_index(drop=True), 'no')

        def plotMultiy2(dftt_newcSUM, xtime):
            # ploting time with Fluid Production (Multiple y-axis)
            userValuesclr = userValues.copy()
            if len(graphNum) != 1:
                if (answer == 'group' or answer == 'both'):

                    if xtime == 'yes':
                        for year in yearsx:
                            plt.axvline(pd.Timestamp(str(year)), color='black', linewidth=1)
                        plot_multi2(dftt_newcSUM, userValuesclr, xtime, figsize=(25, 10));

                        plt.title(str(userValue) + ' Field Cumulative Production');
                        plt.savefig(final_directory + '/' + userValue + ' field cumulative production year multy.png')
                        st.pyplot()

                    else:
                        for tick in np.arange(0, dftt_newcSUM.shape[0] + 1, 12):
                            plt.axvline(tick, color='black', linewidth=1)

                        # df_newcSUM.set_index('Time', inplace=True)
                        plot_multi2(dftt_newcSUM, userValuesclr, xtime, figsize=(25, 10));

                        plt.title(str(userValue) + ' Field Cumulative Production');
                        plt.savefig(final_directory + '/' + userValue + ' field cumulative production month multy.png')
                        st.pyplot()

        # ploting time with Fluid Production (Multiple y-axis)
        plotMultiy2(dftt_newcSUM, 'yes')

        # ploting time with Fluid Production (Multiple y-axis)(indexes)
        if ('OIL' in userValues):
            plotMultiy2(dftt_newcSUM.reset_index(drop=True), 'no')

        # create plots download link
        zipf = zipfile.ZipFile('Group Plots.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('Group Plots', zipf)
        zipf.close()
        st.markdown(get_binary_file_downloader_html('Group Plots.zip', userValue + ' Group Plots'),unsafe_allow_html=True)
