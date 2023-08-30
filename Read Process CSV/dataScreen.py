import os
import numpy as np
from tqdm import tqdm
  
def peakDriftScreen(folderPath, fragmentDict, fragmentMostAbundant, mergedDict, fileExt = '.txt', driftThreshold = 2.5):
    subMassDict = {'d':1.00627674587,'15n':0.997034886,'13c':1.003354835,
                   'unsub':0,'33s':0.999387735,'34s':1.995795825,'36s':3.995009525,
                  '18o':2.0042449924,'17o':1.0042171364}
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]

    for fIdx, fragKey in enumerate(list(fragmentDict.keys())):
        isotopes = fragmentDict[fragKey]
        mostAbundant = fragmentMostAbundant[fIdx]
        for iso in isotopes:
            if iso != "OMIT":
                for fileIdx, (fileKey, mergedDf) in enumerate(mergedDict.items()):
                    observedMassIso = mergedDf[fIdx][mergedDf[fIdx]['mass' + iso]!=0]['mass' + iso].mean()
                    observedMassMostAbundant = mergedDf[fIdx][mergedDf[fIdx]['mass' + mostAbundant]!=0]['mass' + mostAbundant].mean()

                    computedMassMostAbundant = 0
                    mASubs = mostAbundant.split('-')

                    #Find the increase in mass due to substitutions
                    for sub in mASubs:
                        try:
                            computedMassMostAbundant += subMassDict[sub.lower()]
                        except:
                            print("Could not look up substitution " + sub + " correctly.")
                            computedMassMostAbundant += 0

                    computedMassIso = 0
                    mSubs = iso.split('-')
                    #Find the increase in mass due to substitutions
                    for sub in mSubs:
                        try:
                            computedMassIso += subMassDict[sub.lower()]
                        except:
                            print("Could not look up substitution " + sub + " correctly.")
                            computedMassIso += 0

                    #compute observed and theoretical mass differences
                    massDiffActual = computedMassIso - computedMassMostAbundant
                    massDiffObserve = observedMassIso - observedMassMostAbundant

                    peakDrift = np.abs(massDiffObserve - massDiffActual)

                    peakDriftppm = peakDrift / observedMassIso * 10**6

                    if peakDriftppm > driftThreshold:
                        print("Peak Drift Observed for " + fileNames[fileIdx] + ' ' + fragKey + " " + iso + " with size " + str(peakDriftppm))

def RSESNScreen(allOutputDict):
    '''
    Screen all peaks and print any which have RSE/SN > 2

    Inputs:
        allOutputDict: A dictionary containing all of the output ratios, from dataAnalyzerMN.calc_Folder_Output

    Outputs:
        None. Prints flags for peaks that exceed the threshold. 
    '''

    for fragKey, ratioData in allOutputDict[0].items():
        for ratio in ratioData.keys():
            for fileIdx, fileData in enumerate(allOutputDict):

                RSESN = fileData[fragKey][ratio]['RelStError'] / fileData[fragKey][ratio]['ShotNoiseLimit by Quadrature']
                
                if RSESN >= 2:
                    print('File ' + str(fileIdx) + ' ' + fragKey + ' ' + ratio + ' fails RSE/SN Test with value of ' + str(RSESN))
                
def zeroCountsScreen(folderPath, fragmentDict, mergedDict, fileExt = '.txt'):
    '''
    Iterates through all peaks and prints those with zero counts higher than a certain relative threshold.

    Inputs: 
        folderPath: The directory containing the .txt or .csv files from FTStatistic.
        fragmentDict: A dictionary containing information about the fragments and substitutions present in the FTStat output file. 
        mergedDict: A dictionary, where keys are the file names, and the values are lists of dataframes. Each list corresponds to a fragment, and givesthe scans and data for that fragment of that file. 
        fileExt: The file extension of the FTStat output file, either '.txt' or '.csv'
        threshold: The relative number of zero scans to look for

    Outputs:
        None. Prints the name of peaks with more than the threshold number of zero scans. 
    '''
    fileNames = [x for x in os.listdir(folderPath) if x.endswith(fileExt)]

    fragKeys = list(fragmentDict.keys())

    for fragKey, fragIsotopes in fragmentDict.items():
        fragIdx = fragKeys.index(fragKey)
        for iso in fragIsotopes:
            if iso != "OMIT":
                for fileIdx, fileName in enumerate(fileNames):
                    cDf = mergedDict[fileName][fragIdx]
                    thisIsoFileZeros = len(cDf['counts' + iso].values) - np.count_nonzero(cDf['counts' + iso].values)
                    if thisIsoFileZeros > 0:
                        print(fileName + ' ' + iso + ' ' + fragKey + ' has ' + str(thisIsoFileZeros) + ' zero scans, out of ' + str(len(cDf['counts' + iso])) + ' scans (' + str(thisIsoFileZeros / len(cDf['counts' + iso])) + ')')