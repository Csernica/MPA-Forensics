import numpy as np
from scipy import stats

'''
A set of scripts to standardize Orbitrap-IRMS runs
'''

def extractThisRun(extractedData, failedAny, thisIndices, thisRat):
    '''
    set to interact with 'sampleOutputDict', one output from dataAnalyzer_FTStat. Pulls out the 7 values and errors associated with an individual sample and its standards, for a single ratio (e.g., 13C/Unsub)

    Inputs:
        extractedData: A dictionary, like sampleOutputDict from dataAnalyzer_FTStat
        failedAny: A dictionary; keys are ratios ('13C/Unsub') and values are lists of files. If a file is in this list, ignore it.
        thisIndices: A list of the indices of the files in the sampleOutputDict associated with this sample. 
        thisRat: String, like '13C/Unsub'
    '''
    valDict = {'Avg':[],'StdError':[],'RelStdError':[]}


    fileNames = [list(extractedData.keys())[x] for x in thisIndices] # get fileNames
    for fileName in fileNames:
        if fileName in failedAny[thisRat]: #if this file failed, ignore it
            valDict['Avg'].append(None)
            valDict['StdError'].append(None)
            valDict['RelStdError'].append(None)
        else:
            thisData = extractedData[fileName]['96'][thisRat] #otherwise, extract data
            valDict['Avg'].append(thisData['Average'])
            valDict['StdError'].append(thisData['StdError'])
            valDict['RelStdError'].append(thisData['RelStdError'])

    return valDict #Return data

def bracketStandard(thisVals, thisRSE):
    '''
    A routine to standardize using two standard brackets, one immediately preceeding and one immediately following
    
    Inputs: 
        thisVals: A list of 7 values
        thisRSE: A list of 7 relative standard errors.
        
        Both are in the order: Std/Smp/Std/Smp/Std/Smp/Std
        
    Outputs:
        smpStdVals: The standardized sample values
        smpStdErrs: The corresponding errors
    '''

    smpStdVals = []
    smpStdErrs = []

    for i in range(1,7,2):
        if thisVals[i] == None: #if a sample file is bad, just skip it. 
            continue
        elif thisVals[i-1] == None and thisVals[i+1] == None: #if both standards are bad, skip it 
            print("Both standards bad")
            continue
        elif thisVals[i-1] == None: #First standard bad
            smpstd = thisVals[i] / thisVals[i+1]
            smpstdErr = np.sqrt(thisVals[i]**2 + thisVals[i+1]**2)
        elif thisVals[i+1] == None: #Second standard bad
            smpstd = thisVals[i] / thisVals[i-1]
            smpstdErr = np.sqrt(thisVals[i]**2 + thisVals[i-1]**2)
        else: #All files good
            avgStd = 1/2 * (thisVals[i-1] + thisVals[i+1])
            avgStdErr = 1/2 * (thisRSE[i-1] + thisRSE[i+1])
            smpstd = thisVals[i] / avgStd
            smpstdErr = np.sqrt(thisRSE[i]**2 + avgStdErr**2)

        #Add to output list
        smpStdVals.append(smpstd)
        smpStdErrs.append(smpstdErr)

    return np.array(smpStdVals), np.array(smpStdErrs)

def linearStandard(thisVals, thisRSEs):
    '''
    A routine to standardize using a linear fit to all standards
    
    Inputs: 
        thisVals: A list of 7 values
        thisRSE: A list of 7 relative standard errors.
        
        Both are in the order: Std/Smp/Std/Smp/Std/Smp/Std
        
    Outputs:
        smpStdVals: The standardized sample values
        smpStdErrs: The corresponding errors
    '''
    smpStdVals = []
    smpStdErrs = []

    #Pull out good standard values and associated timepoints
    stdxs = []
    stds = []
    for i in range(0,8,2):
        if thisVals[i] == None:
            continue
        else:
            stdxs.append(i / 2)
            stds.append(thisVals[i])

    #Perform a linear regression
    slope, intercept, r_value, p_value, slope_serr = stats.linregress(stdxs,stds)

    #calculate useful intermediates
    n = len(stdxs)
    predictions = slope * np.array(stdxs) + intercept 
    yerr = stds - predictions 
    s_err = np.sum(yerr**2)
    mean_x = np.mean(stdxs)       
    # appropriate t value (where n=4, two tailed 68%)            
    t = stats.t.ppf(1-0.16, n-2)              

    #Define a function to calculate the error in this regression at a certain x value
    def errorAtX(thisX):
        pointErr = np.sqrt(s_err / (n-2)) * np.sqrt(1.0/n + (thisX - mean_x)**2 / np.sum((stdxs-mean_x)**2))
        return pointErr

    #Pull out good sample data and associated timepoints
    smpxs = []
    smps = []
    smpErr = []
    for i in range(1,7,2):
        if thisVals[i] == None:
            continue
        else:
            smpxs.append(i / 2)
            smps.append(thisVals[i])
            smpErr.append(thisRSEs[i])

    smpxs = np.array(smpxs)
    smps = np.array(smps)
    smpErr = np.array(smpErr)

    #Predict values of the standard at these timepoints
    stdPreds = np.array(slope * smpxs + intercept)
    stdRSEPreds = errorAtX(smpxs) / stdPreds

    #Standardize
    smpStdVals = smps / stdPreds
    smpStdErrs = np.sqrt(smpErr**2 + stdRSEPreds**2)

    return np.array(smpStdVals), np.array(smpStdErrs)

def standardizeOneRun(thisVals, thisRSEs, standardize='bracket'):
    '''
    Function wrapper to determine which standardization method to use in an easy way.
    '''
    if standardize == 'bracket':
        return bracketStandard(thisVals, thisRSEs)
    elif standardize == 'linear':
        return linearStandard(thisVals, thisRSEs)
    else:
        raise Exception("Cannot standardize via " + str(standardize))
