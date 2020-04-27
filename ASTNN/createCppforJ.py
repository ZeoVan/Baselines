import pandas as pd
import os

def find_primary_source_file(datapoints):
    """
    Given a list of datapoints representing the files for a single
    testcase, try to find which of the files is the "primary"
    file.
    According to the Juliet documentation, this should be the
    only file which defines the main function.
    In contrast, there is only ever one piece of code in the
    vdisc dataset.
    """

    if len(datapoints) == 1:
        # VDISC case and some of Juliet
        return datapoints.iloc[0]

    elif len(datapoints) > 1:
        # Juliet only case
        for datapoint in datapoints.itertuples():
            for line in datapoint.code.split("\n"):
                if line.startswith("int main("):
                    #primary = datapoint
                    return datapoint

        return datapoints.iloc[0]


def get_bug_lable(testcase, **kwargs):
    """
    Takes in a list of files/datapoints from juliet.csv.zip or
    vdisc_*.csv.gz (as loaded with pandas) matching one particular
    testcase, and preprocesses it ready for the baseline model.
    """
    lable_list = [
        (datapoint.CWE_119, datapoint.CWE_120,datapoint.CWE_469,datapoint.CWE_476,datapoint.CWE_OTHERS)
        for datapoint in testcase.itertuples()
    ]
    for datapoint in testcase.itertuples():
        lable = int(datapoint.CWE_119)+int(datapoint.CWE_120)+int(datapoint.CWE_469)+int(datapoint.CWE_476)+int(datapoint.CWE_OTHERS)
    if lable > 0:
        return "yes"
    else:
        return "no"

if __name__ == '__main__':
    train = pd.read_csv('./juliet_split.csv.gz')
    df = train.groupby(['testcase_ID'])
    for group in df:
        testcase = group[1]   # 取出第二个元素
        #lable = get_bug_lable(testcase)
        primary = find_primary_source_file(testcase)
        fileName = str(primary[1])+"_"+str(primary[0])
        print("allJcpp/"+fileName+".cpp")
        dirname = "allJcpp/"+fileName+"/"
        os.makedirs(dirname)
        flag = False
        for line in primary.code.split("\n"):
            if line.startswith("/* Below is the main("):
                flag = True
            if not line.startswith("/* Below is the main("):
                if not flag:
                    with open(dirname+fileName+".cpp","a+") as f:   
                        f.write(line+"\n")
    '''            
    test = pd.read_csv('./vdisc_test_all.csv') 
    df = test.groupby(['testcase_ID'])
    for group in df:
        testcase = group[1]   # 取出第二个元素
        lable = get_bug_lable(testcase)
        for datapoint in testcase.itertuples():
            code = datapoint.code
            fileName = datapoint.testcase_ID
            pos=code.find('(')
            code = "test_"+lable+code[pos:]
            print("allcpp/"+lable+'_'+fileName+".cpp")
            dirname = "allcpp/"+lable+'_test'+'_'+fileName+"/"
            os.makedirs(dirname)
            with open(dirname+lable+'_test'+'_'+fileName+".cpp","w+") as f:   
                f.write(code)
                
    val = pd.read_csv('./vdisc_validate_all.csv')
    df = val.groupby(['testcase_ID'])
    for group in df:
        testcase = group[1]   # 取出第二个元素
        lable = get_bug_lable(testcase)
        for datapoint in testcase.itertuples():
            code = datapoint.code
            fileName = datapoint.testcase_ID
            pos=code.find('(')
            code = "val_"+lable+code[pos:]
            print("allcpp/"+lable+'_'+fileName+".cpp")
            dirname = "allcpp/"+lable+'_val'+'_'+fileName+"/"
            os.makedirs(dirname)
            with open(dirname+lable+'_val'+'_'+fileName+".cpp","w+") as f:   
                f.write(code)
    '''