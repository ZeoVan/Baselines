import pandas as pd
import os

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
    train = pd.read_csv('./vdisc_train_all.csv')
    df = train.groupby(['testcase_ID'])
    for group in df:
        testcase = group[1]   # 取出第二个元素
        lable = get_bug_lable(testcase)
        for datapoint in testcase.itertuples():
            code = datapoint.code
            fileName = datapoint.testcase_ID
            pos=code.find('(')
            code = "train_"+lable+code[pos:]
            print("allcpp/"+lable+'_'+fileName+".cpp")
            dirname = "allcpp/"+lable+'_train'+'_'+fileName+"/"
            os.makedirs(dirname)
            with open(dirname+lable+'_train'+'_'+fileName+".cpp","w+") as f:   
                f.write(code)
                
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